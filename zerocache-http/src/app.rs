use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use prometheus::{Encoder, IntCounterVec, Opts, Registry, TextEncoder};

use zerocache_core::{normalize_text, reconcile, CacheKey};
use zerocache_ports::{EmbeddingProvider, EmbeddingStore, ProviderError, StoreError};

// A store call is not bounded by PROVIDER_TIMEOUT, which only applies to
// provider HTTP calls -- without this, a stuck store backend (e.g. a stale
// Redis connection) could hang a request indefinitely, and /ready, which
// exists specifically to detect exactly that, would hang right along with
// it, defeating the point of a readiness probe. Bounding every store call
// uniformly here (not just in the Redis adapter) means a hung backend of
// any kind degrades into a fast, visible AppError::Store instead of an
// indefinite hang -- one that would otherwise also block graceful
// shutdown's drain, since with_graceful_shutdown waits for in-flight
// handler futures, and an unbounded store call is awaited inside one.
const STORE_TIMEOUT: Duration = Duration::from_secs(5);

/// Runs a blocking store operation on the blocking thread pool, bounded by
/// `STORE_TIMEOUT`. Note this bounds how long the *caller* waits, not the
/// spawned OS thread itself -- a genuinely wedged sync call still occupies a
/// blocking-pool thread until it eventually returns (Rust cannot preempt
/// blocking code). The Redis adapter additionally sets its own socket-level
/// read/write timeouts, which do bound the underlying call itself; this
/// wrapper is the uniform backstop for any backend, including ones that
/// don't set their own.
///
/// When `task` loops over multiple keys (as `embed_batch`'s reconcile and
/// write-back steps, and `delete_batch`, all do), `STORE_TIMEOUT` bounds the
/// whole loop, not each individual key -- a per-batch budget, not a
/// per-operation one. Fine given `MAX_BATCH_SIZE = 100`, but worth knowing
/// if that constant ever grows.
async fn run_store_task<T, F>(task: F) -> Result<T, AppError>
where
    F: FnOnce() -> Result<T, AppError> + Send + 'static,
    T: Send + 'static,
{
    run_store_task_with_timeout(STORE_TIMEOUT, task).await
}

/// The actual timeout/panic-disambiguation logic, parameterized so tests can
/// exercise the timeout-fires and panic-propagates branches in milliseconds
/// instead of waiting on the real `STORE_TIMEOUT`.
async fn run_store_task_with_timeout<T, F>(timeout: Duration, task: F) -> Result<T, AppError>
where
    F: FnOnce() -> Result<T, AppError> + Send + 'static,
    T: Send + 'static,
{
    tokio::time::timeout(timeout, tokio::task::spawn_blocking(task))
        .await
        .map_err(|_| AppError::Store(StoreError("store operation timed out".into())))?
        .expect("store task panicked")
}

pub struct AppState {
    pub store: Arc<dyn EmbeddingStore>,
    pub providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    pub metrics: Metrics,
}

// Cumulative counters since process start, in Prometheus text-exposition
// format at GET /metrics, labeled by provider. Deliberately minimal — just
// what Phase 1's success criteria (measured hit rate, measured tokens
// billed) need, not the full PRD §11 spec. No owner/tenant label: that would
// leak tenant identity into a monitoring system and create unbounded
// cardinality (one series per tenant). Provider is small and bounded.
pub struct Metrics {
    registry: Registry,
    hits: IntCounterVec,
    misses: IntCounterVec,
    prompt_tokens_billed: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let hits = IntCounterVec::new(
            Opts::new(
                "zerocache_cache_hits_total",
                "Total embedding requests served from cache without a provider call",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let misses = IntCounterVec::new(
            Opts::new(
                "zerocache_cache_misses_total",
                "Total embedding requests that required a provider call",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let prompt_tokens_billed = IntCounterVec::new(
            Opts::new(
                "zerocache_provider_prompt_tokens_total",
                "Total prompt tokens billed by the embedding provider",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");

        registry
            .register(Box::new(hits.clone()))
            .expect("registering a metric once on a fresh registry cannot fail");
        registry
            .register(Box::new(misses.clone()))
            .expect("registering a metric once on a fresh registry cannot fail");
        registry
            .register(Box::new(prompt_tokens_billed.clone()))
            .expect("registering a metric once on a fresh registry cannot fail");

        Self { registry, hits, misses, prompt_tokens_billed }
    }

    fn record(&self, provider: &str, stats: &BatchStats) {
        self.hits.with_label_values(&[provider]).inc_by(stats.hits as u64);
        self.misses.with_label_values(&[provider]).inc_by(stats.misses as u64);
        self.prompt_tokens_billed
            .with_label_values(&[provider])
            .inc_by(stats.provider_prompt_tokens as u64);
    }

    /// Renders all registered metrics in Prometheus text exposition format.
    pub fn encode(&self) -> String {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&metric_families, &mut buffer)
            .expect("encoding already-gathered metrics cannot fail");
        String::from_utf8(buffer).expect("prometheus text encoder always emits valid utf8")
    }
}

#[derive(Debug)]
pub enum AppError {
    Store(StoreError),
    Provider(ProviderError),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Store(e) => write!(f, "{e}"),
            AppError::Provider(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AppError {}

pub struct BatchStats {
    pub hits: usize,
    pub misses: usize,
    pub provider_prompt_tokens: u32,
    pub provider_total_tokens: u32,
}

/// Everything one `/{provider}/v1/embeddings` call needs to run the cache
/// flow: which adapter to call, what to call it (for the cache key and the
/// metrics label — kept separate from the adapter reference itself since the
/// registry is keyed by name, not by adapter identity), whose namespace this
/// is, and the batch itself.
pub struct EmbedRequest<'a> {
    pub provider: &'a dyn EmbeddingProvider,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub texts: &'a [String],
}

/// Runs the cache flow for one batch: reconcile against the store (scoped to
/// `request.owner_id` + `request.provider_name` via the cache key), fetch
/// only the misses from `request.provider`, write them back, and return
/// vectors in the same order as `request.texts`.
///
/// Text is normalized (trimmed, internal whitespace collapsed) before both
/// hashing and being sent to the provider — applying it in only one place
/// would let the cache key and what's actually embedded silently diverge.
/// Misses that normalize to an identical string are deduplicated: the
/// provider is called once per unique text, not once per original index, and
/// the same fetched vector is broadcast back to every matching position.
///
/// Store reads and store writes each run via `run_store_task`, which puts
/// them on their own `tokio::task::spawn_blocking` (since `sled`/`redis` are
/// synchronous) bounded by `STORE_TIMEOUT`. The provider call in between is
/// awaited directly — it's async now, not blocking, so it runs on the same
/// runtime as the rest of the server instead of needing a thread-pool hop,
/// and is bounded separately by each adapter's own `PROVIDER_TIMEOUT`.
///
/// A store or provider failure aborts the whole batch rather than degrading
/// silently into an extra miss or a dropped write — the caller is expected
/// to surface it as an error response.
pub async fn embed_batch(state: &AppState, request: EmbedRequest<'_>) -> Result<(Vec<Vec<f32>>, BatchStats), AppError> {
    let provider_version = request.provider.version();
    let normalized_texts: Vec<String> = request.texts.iter().map(|text| normalize_text(text)).collect();
    let keys: Vec<CacheKey> = normalized_texts
        .iter()
        .map(|text| CacheKey::derive(request.owner_id, request.provider_name, request.model, provider_version, text))
        .collect();

    let reconciled = {
        let store = Arc::clone(&state.store);
        let keys_for_lookup = keys.clone();
        run_store_task(move || reconcile(&keys_for_lookup, |key| store.get(key).map_err(AppError::Store))).await?
    };

    let hits = reconciled.hits.len();
    let misses = reconciled.misses.len();

    let mut results: Vec<Option<Vec<f32>>> = vec![None; request.texts.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    let mut provider_prompt_tokens = 0;
    let mut provider_total_tokens = 0;

    if !reconciled.misses.is_empty() {
        // Dedupe misses by CacheKey: several original indices can normalize
        // to the identical text (or already be literal duplicates in the
        // input), and each such group shares one CacheKey. Fetch each
        // unique text once, then broadcast its vector to every index that
        // shares its key.
        let mut unique_miss_texts: Vec<String> = Vec::new();
        let mut key_to_unique_index: HashMap<CacheKey, usize> = HashMap::new();
        let mut index_to_unique_index: Vec<(usize, usize)> = Vec::with_capacity(reconciled.misses.len());

        for (index, key) in &reconciled.misses {
            let unique_index = *key_to_unique_index.entry(*key).or_insert_with(|| {
                unique_miss_texts.push(normalized_texts[*index].clone());
                unique_miss_texts.len() - 1
            });
            index_to_unique_index.push((*index, unique_index));
        }

        let (fetched, usage) = request
            .provider
            .embed_batch(request.api_key, request.model, &unique_miss_texts)
            .await
            .map_err(AppError::Provider)?;
        provider_prompt_tokens = usage.prompt_tokens;
        provider_total_tokens = usage.total_tokens;

        for (index, unique_index) in &index_to_unique_index {
            results[*index] = Some(fetched[*unique_index].clone());
        }

        let mut writes = Vec::with_capacity(key_to_unique_index.len());
        for (key, unique_index) in key_to_unique_index {
            writes.push((key, fetched[unique_index].clone()));
        }

        let store = Arc::clone(&state.store);
        run_store_task(move || -> Result<(), AppError> {
            for (key, vector) in writes {
                store.put(key, vector).map_err(AppError::Store)?;
            }
            Ok(())
        })
        .await?;
    }

    let vectors = results
        .into_iter()
        .map(|v| v.expect("every index must be filled by a hit or a miss"))
        .collect();

    let stats = BatchStats { hits, misses, provider_prompt_tokens, provider_total_tokens };
    state.metrics.record(request.provider_name, &stats);

    Ok((vectors, stats))
}

/// Everything one `DELETE /{provider}/v1/embeddings` call needs: which
/// entries to remove, scoped to the caller's own namespace via `owner_id`
/// exactly like a normal read/write request — a caller can only ever delete
/// entries in their own namespace, never anyone else's. `provider` is only
/// used for its `version()`, to compute the same keys a matching `POST`
/// would have used; no provider call is ever made for a delete.
pub struct DeleteRequest<'a> {
    pub provider: &'a dyn EmbeddingProvider,
    pub provider_name: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub texts: &'a [String],
}

/// Deletes the cache entries that a matching `POST` with the same
/// provider/model/texts (and the same caller) would have hit. Returns the
/// number of keys the delete was attempted for -- deletion is idempotent, so
/// this is not "how many existed," just "how many were requested."
pub async fn delete_batch(state: &AppState, request: DeleteRequest<'_>) -> Result<usize, AppError> {
    let provider_version = request.provider.version();
    let keys: Vec<CacheKey> = request
        .texts
        .iter()
        .map(|text| CacheKey::derive(request.owner_id, request.provider_name, request.model, provider_version, &normalize_text(text)))
        .collect();

    let count = keys.len();
    let store = Arc::clone(&state.store);
    run_store_task(move || -> Result<(), AppError> {
        for key in keys {
            store.delete(&key).map_err(AppError::Store)?;
        }
        Ok(())
    })
    .await?;

    Ok(count)
}

/// A fixed, reserved cache key used only to prove the store is reachable --
/// never a real cache entry a caller could hit or collide with, since no
/// real request can produce owner_id = [0u8; 32] (a real owner_id is always
/// derived from a hashed API key, which blake3 never maps to all-zero
/// output in practice) combined with this reserved provider/model string.
fn readiness_check_key() -> CacheKey {
    CacheKey::derive([0u8; 32], "__zerocache_internal__", "__readiness_check__", "v1", "")
}

/// Proves the configured store backend is actually reachable, not just
/// that the process is running. A miss is a healthy, expected result (the
/// sentinel key is never written to) -- only a store-level error means
/// "not ready."
pub async fn check_store_readiness(state: &AppState) -> Result<(), AppError> {
    let store = Arc::clone(&state.store);
    let key = readiness_check_key();
    run_store_task(move || store.get(&key).map(|_| ()).map_err(AppError::Store)).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdHashMap;
    use std::sync::Mutex;

    use zerocache_ports::ProviderUsage;

    use super::*;

    struct MockStore {
        data: Mutex<StdHashMap<CacheKey, Vec<f32>>>,
    }

    impl MockStore {
        fn empty() -> Self {
            Self { data: Mutex::new(StdHashMap::new()) }
        }

        fn with(entries: Vec<(CacheKey, Vec<f32>)>) -> Self {
            Self { data: Mutex::new(entries.into_iter().collect()) }
        }
    }

    impl EmbeddingStore for MockStore {
        fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }

        fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError> {
            self.data.lock().unwrap().insert(key, vector);
            Ok(())
        }

        fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }
    }

    struct FailingStore;

    impl EmbeddingStore for FailingStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Err(StoreError("mock store failure".into()))
        }

        fn put(&self, _key: CacheKey, _vector: Vec<f32>) -> Result<(), StoreError> {
            Err(StoreError("mock store failure".into()))
        }

        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Err(StoreError("mock store failure".into()))
        }
    }

    struct MockProvider {
        response: Vec<Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            assert_eq!(texts.len(), self.response.len(), "mock provider called with unexpected batch size");
            Ok((self.response.clone(), ProviderUsage { prompt_tokens: 10, total_tokens: 10 }))
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }
    }

    struct FailingProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for FailingProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            Err(ProviderError("mock provider failure".into()))
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }
    }

    struct PanicProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for PanicProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            panic!("provider must not be called when the store lookup already failed");
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }
    }

    const OWNER_A: [u8; 32] = [1u8; 32];
    const OWNER_B: [u8; 32] = [2u8; 32];

    fn state_with(store: impl EmbeddingStore + 'static) -> AppState {
        AppState {
            store: Arc::new(store),
            providers: StdHashMap::new(),
            metrics: Metrics::new(),
        }
    }

    #[tokio::test]
    async fn all_hits_skips_provider_entirely() {
        let key = CacheKey::derive(OWNER_A, "openai", "m", "mock-v1", "cached text");
        let state = state_with(MockStore::with(vec![(key, vec![1.0, 2.0])]));
        let provider = PanicProvider;
        let texts = vec!["cached text".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.provider_prompt_tokens, 0);
    }

    #[tokio::test]
    async fn all_misses_calls_provider_and_writes_back_to_store() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider { response: vec![vec![9.0, 9.0]] };
        let texts = vec!["fresh text".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![9.0, 9.0]]);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 10);

        let key = CacheKey::derive(OWNER_A, "openai", "m", "mock-v1", "fresh text");
        assert_eq!(state.store.get(&key).unwrap(), Some(vec![9.0, 9.0]));
    }

    #[tokio::test]
    async fn mixed_batch_preserves_original_order() {
        let hit_key = CacheKey::derive(OWNER_A, "openai", "m", "mock-v1", "old");
        let state = state_with(MockStore::with(vec![(hit_key, vec![1.0])]));
        let provider = MockProvider { response: vec![vec![2.0]] };
        let texts = vec!["new".to_string(), "old".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![2.0], vec![1.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn different_owners_never_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider { response: vec![vec![7.0]] };
        let texts = vec!["identical text".to_string()];

        let (_, stats_a) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key-a",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_a.misses, 1, "owner A's first request must be a miss");

        let (_, stats_b) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key-b",
                owner_id: OWNER_B,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_b.misses, 1, "owner B must not hit owner A's cache entry");
    }

    #[tokio::test]
    async fn store_lookup_failure_aborts_before_any_provider_call() {
        let state = state_with(FailingStore);
        let provider = PanicProvider;
        let texts = vec!["text".to_string()];

        let result = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[tokio::test]
    async fn provider_failure_surfaces_as_app_error() {
        let state = state_with(MockStore::empty());
        let provider = FailingProvider;
        let texts = vec!["text".to_string()];

        let result = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Provider(_))));
    }

    #[tokio::test]
    async fn metrics_are_labeled_by_provider_and_only_recorded_on_success() {
        let state = state_with(MockStore::empty());
        let openai_provider = MockProvider { response: vec![vec![1.0]] };
        let texts = vec!["text".to_string()];

        embed_batch(
            &state,
            EmbedRequest {
                provider: &openai_provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        let mistral_texts = vec!["other text".to_string()];
        let failing_provider = FailingProvider;
        let _ = embed_batch(
            &state,
            EmbedRequest {
                provider: &failing_provider,
                provider_name: "mistral",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &mistral_texts,
            },
        )
        .await;

        let metrics_text = state.metrics.encode();
        assert!(metrics_text.contains("zerocache_cache_misses_total{provider=\"openai\"} 1"));
        assert!(
            !metrics_text.contains("zerocache_cache_misses_total{provider=\"mistral\"}"),
            "a failed request must not record any metric for that provider"
        );
    }

    #[tokio::test]
    async fn different_provider_versions_produce_different_cache_entries() {
        struct VersionedProvider {
            version: &'static str,
            response: Vec<Vec<f32>>,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for VersionedProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                assert_eq!(texts.len(), self.response.len());
                Ok((self.response.clone(), ProviderUsage::default()))
            }

            fn version(&self) -> &'static str {
                self.version
            }
        }

        let state = state_with(MockStore::empty());
        let texts = vec!["same text".to_string()];

        let v1 = VersionedProvider { version: "1.0.0", response: vec![vec![1.0]] };
        let (_, stats_v1) = embed_batch(
            &state,
            EmbedRequest {
                provider: &v1,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_v1.misses, 1);

        let v2 = VersionedProvider { version: "2.0.0", response: vec![vec![2.0]] };
        let (_, stats_v2) = embed_batch(
            &state,
            EmbedRequest {
                provider: &v2,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_v2.misses, 1, "a different adapter version must not hit the old version's cache entry");
    }

    #[tokio::test]
    async fn duplicate_texts_in_one_batch_call_provider_only_once() {
        struct CountingProvider {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for CountingProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(texts.len(), 1, "duplicate texts in the batch must be deduplicated before calling the provider");
                Ok((vec![vec![42.0]], ProviderUsage::default()))
            }

            fn version(&self) -> &'static str {
                "mock-v1"
            }
        }

        let state = state_with(MockStore::empty());
        let provider = CountingProvider { call_count: std::sync::atomic::AtomicUsize::new(0) };
        // Same text three times in one batch.
        let texts = vec!["same text".to_string(), "same text".to_string(), "same text".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(provider.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(vectors, vec![vec![42.0], vec![42.0], vec![42.0]], "the same vector must be broadcast to every duplicate position");
        assert_eq!(stats.misses, 3, "hit/miss accounting still reflects all three original positions");
    }

    #[tokio::test]
    async fn whitespace_variants_of_the_same_text_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider { response: vec![vec![1.0]] };
        let texts = vec!["  hello   world  ".to_string()];

        embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        // A second request with different incidental whitespace, but the
        // same normalized content, must hit the entry the first request
        // wrote -- proving the store key was derived from normalized text.
        let panic_provider = PanicProvider;
        let differently_spaced_texts = vec!["hello world".to_string()];
        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &panic_provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &differently_spaced_texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(stats.hits, 1);
        assert_eq!(vectors, vec![vec![1.0]]);
    }

    #[tokio::test]
    async fn delete_batch_removes_entries_so_a_later_request_misses_again() {
        let key = CacheKey::derive(OWNER_A, "openai", "m", "mock-v1", "to be forgotten");
        let state = state_with(MockStore::with(vec![(key, vec![9.0]) ]));
        let provider = MockProvider { response: vec![vec![1.0]] };
        let texts = vec!["to be forgotten".to_string()];

        let deleted = delete_batch(
            &state,
            DeleteRequest {
                provider: &provider,
                provider_name: "openai",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(deleted, 1);

        let (_, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: &provider,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats.hits, 0, "a deleted entry must be a miss again, not still cached");
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn delete_batch_failure_surfaces_as_app_error() {
        let state = state_with(FailingStore);
        let provider = PanicProvider;
        let texts = vec!["text".to_string()];

        let result = delete_batch(
            &state,
            DeleteRequest {
                provider: &provider,
                provider_name: "openai",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[tokio::test]
    async fn readiness_check_succeeds_against_a_healthy_store() {
        let state = state_with(MockStore::empty());
        assert!(check_store_readiness(&state).await.is_ok());
    }

    #[tokio::test]
    async fn readiness_check_fails_when_the_store_is_unreachable() {
        let state = state_with(FailingStore);
        assert!(matches!(check_store_readiness(&state).await, Err(AppError::Store(_))));
    }

    #[tokio::test]
    async fn run_store_task_times_out_on_a_hung_closure() {
        let result: Result<(), AppError> = run_store_task_with_timeout(Duration::from_millis(20), || {
            std::thread::sleep(Duration::from_secs(5));
            Ok(())
        })
        .await;

        assert!(matches!(result, Err(AppError::Store(_))), "a closure that outlives the timeout must surface as a store timeout error");
    }

    #[tokio::test]
    #[should_panic(expected = "store task panicked")]
    async fn run_store_task_propagates_a_panic_rather_than_reporting_it_as_a_timeout() {
        // A generous timeout, so if this test ever fails by reporting a
        // timeout instead of panicking, that's a real regression -- the
        // panic resolves almost instantly, long before the timeout could
        // fire on its own.
        let _: Result<(), AppError> = run_store_task_with_timeout(Duration::from_secs(30), || {
            panic!("simulated store task panic");
        })
        .await;
    }
}

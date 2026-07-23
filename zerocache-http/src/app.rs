use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use prometheus::{Encoder, IntCounterVec, Opts, Registry, TextEncoder};

use zerocache_core::{reconcile, CacheKey};
use zerocache_ports::{EmbeddingProvider, EmbeddingStore, ProviderError, StoreError};

pub struct AppState {
    pub store: Arc<dyn EmbeddingStore>,
    pub providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    // Bumped when an adapter's handling of a model changes in a way that
    // could change its output, independent of what the client sends as `model`.
    pub model_version: String,
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
/// vectors in the same order as `request.texts`. Store/provider calls are
/// blocking, so callers on the async server must run this inside
/// `tokio::task::spawn_blocking`.
///
/// A store or provider failure aborts the whole batch rather than degrading
/// silently into an extra miss or a dropped write — the caller is expected
/// to surface it as an error response.
pub fn embed_batch(state: &AppState, request: EmbedRequest) -> Result<(Vec<Vec<f32>>, BatchStats), AppError> {
    let keys: Vec<CacheKey> = request
        .texts
        .iter()
        .map(|text| {
            CacheKey::derive(request.owner_id, request.provider_name, request.model, &state.model_version, text)
        })
        .collect();

    let reconciled = reconcile(&keys, |key| state.store.get(key).map_err(AppError::Store))?;
    let hits = reconciled.hits.len();
    let misses = reconciled.misses.len();

    let mut results: Vec<Option<Vec<f32>>> = vec![None; request.texts.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    let mut provider_prompt_tokens = 0;
    let mut provider_total_tokens = 0;

    if !reconciled.misses.is_empty() {

        let miss_texts: Vec<String> = reconciled
            .misses
            .iter()
            .map(|(index, _)| request.texts[*index].clone())
            .collect();
        let (fetched, usage) = request
            .provider
            .embed_batch(request.api_key, request.model, &miss_texts)
            .map_err(AppError::Provider)?;
        provider_prompt_tokens = usage.prompt_tokens;
        provider_total_tokens = usage.total_tokens;

        for ((index, key), vector) in reconciled.misses.into_iter().zip(fetched.into_iter()) {
            state.store.put(key, vector.clone()).map_err(AppError::Store)?;
            results[index] = Some(vector);
        }
    }

    let vectors = results
        .into_iter()
        .map(|v| v.expect("every index must be filled by a hit or a miss"))
        .collect();

    let stats = BatchStats { hits, misses, provider_prompt_tokens, provider_total_tokens };
    state.metrics.record(request.provider_name, &stats);

    Ok((vectors, stats))
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
    }

    struct FailingStore;

    impl EmbeddingStore for FailingStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Err(StoreError("mock store failure".into()))
        }

        fn put(&self, _key: CacheKey, _vector: Vec<f32>) -> Result<(), StoreError> {
            Err(StoreError("mock store failure".into()))
        }
    }

    struct MockProvider {
        response: Vec<Vec<f32>>,
    }

    impl EmbeddingProvider for MockProvider {
        fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            assert_eq!(texts.len(), self.response.len(), "mock provider called with unexpected batch size");
            Ok((self.response.clone(), ProviderUsage { prompt_tokens: 10, total_tokens: 10 }))
        }
    }

    struct FailingProvider;

    impl EmbeddingProvider for FailingProvider {
        fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            Err(ProviderError("mock provider failure".into()))
        }
    }

    struct PanicProvider;

    impl EmbeddingProvider for PanicProvider {
        fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            panic!("provider must not be called when the store lookup already failed");
        }
    }

    const OWNER_A: [u8; 32] = [1u8; 32];
    const OWNER_B: [u8; 32] = [2u8; 32];

    fn state_with(store: impl EmbeddingStore + 'static) -> AppState {
        AppState {
            store: Arc::new(store),
            providers: StdHashMap::new(),
            model_version: "v1".to_string(),
            metrics: Metrics::new(),
        }
    }

    #[test]
    fn all_hits_skips_provider_entirely() {
        let key = CacheKey::derive(OWNER_A, "openai", "m", "v1", "cached text");
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
        .unwrap();

        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.provider_prompt_tokens, 0);
    }

    #[test]
    fn all_misses_calls_provider_and_writes_back_to_store() {
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
        .unwrap();

        assert_eq!(vectors, vec![vec![9.0, 9.0]]);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 10);

        let key = CacheKey::derive(OWNER_A, "openai", "m", "v1", "fresh text");
        assert_eq!(state.store.get(&key).unwrap(), Some(vec![9.0, 9.0]));
    }

    #[test]
    fn mixed_batch_preserves_original_order() {
        let hit_key = CacheKey::derive(OWNER_A, "openai", "m", "v1", "old");
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
        .unwrap();

        assert_eq!(vectors, vec![vec![2.0], vec![1.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn different_owners_never_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider { response: vec![vec![7.0]] };
        let texts = vec!["identical text".to_string()];

        // Owner A misses, gets cached under owner A's namespace.
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
        .unwrap();
        assert_eq!(stats_a.misses, 1, "owner A's first request must be a miss");

        // Owner B, same provider/model/text, different owner_id: must also
        // be a miss, not a free ride off owner A's cached entry.
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
        .unwrap();
        assert_eq!(stats_b.misses, 1, "owner B must not hit owner A's cache entry");
    }

    #[test]
    fn store_lookup_failure_aborts_before_any_provider_call() {
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
        );

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[test]
    fn provider_failure_surfaces_as_app_error() {
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
        );

        assert!(matches!(result, Err(AppError::Provider(_))));
    }

    #[test]
    fn metrics_are_labeled_by_provider_and_only_recorded_on_success() {
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
        );

        let metrics_text = state.metrics.encode();
        assert!(metrics_text.contains("zerocache_cache_misses_total{provider=\"openai\"} 1"));
        assert!(
            !metrics_text.contains("zerocache_cache_misses_total{provider=\"mistral\"}"),
            "a failed request must not record any metric for that provider"
        );
    }
}

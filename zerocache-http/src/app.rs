use std::fmt;
use std::sync::Arc;

use prometheus::{Encoder, IntCounter, Registry, TextEncoder};

use zerocache_core::{reconcile, CacheKey};
use zerocache_ports::{EmbeddingProvider, EmbeddingStore, ProviderError, StoreError};

pub struct AppState {
    pub store: Arc<dyn EmbeddingStore>,
    pub provider: Arc<dyn EmbeddingProvider>,
    // Bumped when the adapter's handling of a model changes in a way that
    // could change its output, independent of what the client sends as `model`.
    pub model_version: String,
    pub metrics: Metrics,
}

// Cumulative counters since process start, in Prometheus text-exposition
// format at GET /metrics. Deliberately minimal — just what Phase 1's success
// criteria (measured hit rate, measured tokens billed) need, not the full
// PRD §11 spec (per-consumer tagging, latency-saved-vs-baseline), which
// needs decisions not yet made. Chosen over a bespoke JSON /stats endpoint
// because with multiple replicas behind Redis, per-pod counters only mean
// something once something aggregates across pods — which is exactly what
// a Prometheus scrape-and-sum does and a single pod's JSON response can't.
pub struct Metrics {
    registry: Registry,
    hits: IntCounter,
    misses: IntCounter,
    prompt_tokens_billed: IntCounter,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let hits = IntCounter::new(
            "zerocache_cache_hits_total",
            "Total embedding requests served from cache without a provider call",
        )
        .expect("metric name/help are hardcoded and valid");
        let misses = IntCounter::new(
            "zerocache_cache_misses_total",
            "Total embedding requests that required a provider call",
        )
        .expect("metric name/help are hardcoded and valid");
        let prompt_tokens_billed = IntCounter::new(
            "zerocache_provider_prompt_tokens_total",
            "Total prompt tokens billed by the embedding provider",
        )
        .expect("metric name/help are hardcoded and valid");

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

    fn record(&self, stats: &BatchStats) {
        self.hits.inc_by(stats.hits as u64);
        self.misses.inc_by(stats.misses as u64);
        self.prompt_tokens_billed.inc_by(stats.provider_prompt_tokens as u64);
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

/// Runs the cache flow for one batch: reconcile against the store, fetch only
/// the misses from the provider, write them back, and return vectors in the
/// same order as `texts`. Store/provider calls are blocking, so callers on
/// the async server must run this inside `tokio::task::spawn_blocking`.
///
/// A store or provider failure aborts the whole batch rather than degrading
/// silently into an extra miss or a dropped write — the caller is expected
/// to surface it as an error response.
pub fn embed_batch(
    state: &AppState,
    model: &str,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, BatchStats), AppError> {
    let keys: Vec<CacheKey> = texts
        .iter()
        .map(|text| CacheKey::derive(model, &state.model_version, text))
        .collect();

    let reconciled = reconcile(&keys, |key| state.store.get(key).map_err(AppError::Store))?;
    let hits = reconciled.hits.len();
    let misses = reconciled.misses.len();

    let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    let mut provider_prompt_tokens = 0;
    let mut provider_total_tokens = 0;

    if !reconciled.misses.is_empty() {
        let miss_texts: Vec<String> = reconciled
            .misses
            .iter()
            .map(|(index, _)| texts[*index].clone())
            .collect();
        let (fetched, usage) = state
            .provider
            .embed_batch(model, &miss_texts)
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
    state.metrics.record(&stats);

    Ok((vectors, stats))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use zerocache_ports::ProviderUsage;

    use super::*;

    struct MockStore {
        data: Mutex<HashMap<CacheKey, Vec<f32>>>,
    }

    impl MockStore {
        fn empty() -> Self {
            Self { data: Mutex::new(HashMap::new()) }
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
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            panic!("provider must not be called when the store lookup already failed");
        }
    }

    fn state_with(store: impl EmbeddingStore + 'static, provider: impl EmbeddingProvider + 'static) -> AppState {
        AppState {
            store: Arc::new(store),
            provider: Arc::new(provider),
            model_version: "v1".to_string(),
            metrics: Metrics::new(),
        }
    }

    #[test]
    fn all_hits_skips_provider_entirely() {
        let key = CacheKey::derive("m", "v1", "cached text");
        let state = state_with(MockStore::with(vec![(key, vec![1.0, 2.0])]), PanicProvider);

        let (vectors, stats) = embed_batch(&state, "m", &["cached text".to_string()]).unwrap();

        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.provider_prompt_tokens, 0);
    }

    #[test]
    fn all_misses_calls_provider_and_writes_back_to_store() {
        let store = MockStore::empty();
        let provider = MockProvider { response: vec![vec![9.0, 9.0]] };
        let state = state_with(store, provider);

        let (vectors, stats) = embed_batch(&state, "m", &["fresh text".to_string()]).unwrap();

        assert_eq!(vectors, vec![vec![9.0, 9.0]]);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 10);

        let key = CacheKey::derive("m", "v1", "fresh text");
        assert_eq!(state.store.get(&key).unwrap(), Some(vec![9.0, 9.0]));
    }

    #[test]
    fn mixed_batch_preserves_original_order() {
        let hit_key = CacheKey::derive("m", "v1", "old");
        let store = MockStore::with(vec![(hit_key, vec![1.0])]);
        let provider = MockProvider { response: vec![vec![2.0]] };
        let state = state_with(store, provider);

        let texts = vec!["new".to_string(), "old".to_string()];
        let (vectors, stats) = embed_batch(&state, "m", &texts).unwrap();

        assert_eq!(vectors, vec![vec![2.0], vec![1.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn store_lookup_failure_aborts_before_any_provider_call() {
        let state = state_with(FailingStore, PanicProvider);

        let result = embed_batch(&state, "m", &["text".to_string()]);

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[test]
    fn provider_failure_surfaces_as_app_error() {
        let state = state_with(MockStore::empty(), FailingProvider);

        let result = embed_batch(&state, "m", &["text".to_string()]);

        assert!(matches!(result, Err(AppError::Provider(_))));
    }

    #[test]
    fn metrics_are_recorded_on_success_but_not_on_failure() {
        let success_state = state_with(MockStore::empty(), MockProvider { response: vec![vec![1.0]] });
        embed_batch(&success_state, "m", &["text".to_string()]).unwrap();
        assert!(success_state.metrics.encode().contains("zerocache_cache_misses_total 1"));

        let failure_state = state_with(MockStore::empty(), FailingProvider);
        let _ = embed_batch(&failure_state, "m", &["text".to_string()]);
        assert!(failure_state.metrics.encode().contains("zerocache_cache_misses_total 0"));
    }
}

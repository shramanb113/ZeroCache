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

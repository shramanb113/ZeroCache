use std::sync::Arc;

use zerocache_core::{reconcile, CacheKey};
use zerocache_ports::{EmbeddingProvider, EmbeddingStore};

pub struct AppState {
    pub store: Arc<dyn EmbeddingStore>,
    pub provider: Arc<dyn EmbeddingProvider>,
    // Bumped when the adapter's handling of a model changes in a way that
    // could change its output, independent of what the client sends as `model`.
    pub model_version: String,
}

/// Runs the cache flow for one batch: reconcile against the store, fetch only
/// the misses from the provider, write them back, and return vectors in the
/// same order as `texts`. Store/provider calls are blocking, so callers on
/// the async server must run this inside `tokio::task::spawn_blocking`.
pub fn embed_batch(state: &AppState, model: &str, texts: &[String]) -> Vec<Vec<f32>> {
    let keys: Vec<CacheKey> = texts
        .iter()
        .map(|text| CacheKey::derive(model, &state.model_version, text))
        .collect();

    let reconciled = reconcile(&keys, |key| state.store.get(key));

    let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    if !reconciled.misses.is_empty() {
        let miss_texts: Vec<String> = reconciled
            .misses
            .iter()
            .map(|(index, _)| texts[*index].clone())
            .collect();
        let fetched = state.provider.embed_batch(model, &miss_texts);

        for ((index, key), vector) in reconciled.misses.into_iter().zip(fetched.into_iter()) {
            state.store.put(key, vector.clone());
            results[index] = Some(vector);
        }
    }

    results
        .into_iter()
        .map(|v| v.expect("every index must be filled by a hit or a miss"))
        .collect()
}

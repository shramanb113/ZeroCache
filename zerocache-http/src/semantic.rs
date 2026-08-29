//! Feature-gated helpers for the semantic completion tier. `complete()` calls
//! `semantic_inputs` + `semantic_probe` on an exact-match miss and
//! `record_vector` after a 2xx write-back; `main` calls `build_semantic_state`
//! + `rebuild_index` at startup.

use std::sync::Arc;

use tracing::warn;

use zerocache_core::{coarse_key_hash, completion_fuzzy_text, CacheKey, MatchUnit};
use zerocache_ports::{CompletionStore, CompletionVectorStore, VectorRecord};
use zerocache_semantic::{ScopeKey, SemanticIndex, TextEmbed, TextEmbedder};

use crate::app::{run_store_task, AppError};
use crate::config::Config;

/// Bump only when the coarse-canonical *format* changes. `rebuild_index` skips
/// persisted records with a different value.
pub const SEMANTIC_INDEX_VERSION: u8 = 1;

pub struct SemanticState {
    pub embedder: Arc<dyn TextEmbed>,
    pub index: Arc<SemanticIndex>,
    pub vector_store: Arc<dyn CompletionVectorStore>,
    pub threshold: f32,
    pub match_unit: MatchUnit,
}

pub struct SemanticHit {
    pub record_bytes: Vec<u8>,
    pub score: f32,
}

pub fn scope_key(owner_id: &[u8; 32], provider: &str, cache_scope: &str, model: &str) -> ScopeKey {
    let mut h = blake3::Hasher::new();
    h.update(owner_id);
    h.update(b"\0");
    h.update(provider.as_bytes());
    h.update(b"\0");
    h.update(cache_scope.as_bytes());
    h.update(b"\0");
    h.update(model.as_bytes());
    *h.finalize().as_bytes()
}

/// The fuzzy text to embed and the coarse-key hash to gate on, or `None` when
/// the request has no embeddable span.
pub fn semantic_inputs(match_unit: MatchUnit, body: &serde_json::Value) -> Option<(String, [u8; 32])> {
    let text = completion_fuzzy_text(body, match_unit)?;
    Some((text, coarse_key_hash(body, match_unit)))
}

/// Search for a near neighbour that also matches `coarse_hash` exactly and
/// clears the threshold, then fetch its stored completion. A neighbour whose
/// blob is gone (TTL) is tombstoned + its record deleted, and the result is
/// `Ok(None)`.
pub async fn semantic_probe(
    sem: &SemanticState,
    completion_store: &Arc<dyn CompletionStore>,
    scope: ScopeKey,
    coarse_hash: [u8; 32],
    qvec: &[f32],
) -> Result<Option<SemanticHit>, AppError> {
    let Some(hit) = sem.index.search(scope, qvec, coarse_hash, sem.threshold) else {
        return Ok(None);
    };
    let exact_key = hit.exact_key;
    let store = Arc::clone(completion_store);
    let bytes = run_store_task(move || store.get(&exact_key).map_err(AppError::Store)).await?;

    match bytes {
        Some(record_bytes) => Ok(Some(SemanticHit {
            record_bytes,
            score: hit.score,
        })),
        None => {
            sem.index.tombstone(scope, &exact_key);
            if let Err(e) = sem.vector_store.delete(&exact_key) {
                warn!("semantic: failed to drop a stale vector record: {e}");
            }
            Ok(None)
        }
    }
}

/// Best-effort: persist and index the query vector for a just-stored
/// completion. Never fails the request.
pub async fn record_vector(
    sem: &SemanticState,
    exact_key: CacheKey,
    scope: ScopeKey,
    coarse_hash: [u8; 32],
    qvec: Vec<f32>,
) {
    let record = VectorRecord {
        exact_key,
        scope_hash: scope,
        coarse_key_hash: coarse_hash,
        index_version: SEMANTIC_INDEX_VERSION,
        vector: qvec.clone(),
    };
    let vs = Arc::clone(&sem.vector_store);
    match run_store_task(move || vs.insert(record).map_err(AppError::Store)).await {
        Ok(()) => sem.index.insert(scope, &qvec, coarse_hash, exact_key),
        Err(e) => warn!("semantic: failed to persist a vector record, not indexing it: {e}"),
    }
}

/// Build the semantic state, or `None` when disabled. Exits the process on
/// embedder-load failure — the operator asked for the tier and a silent
/// downgrade would hide a broken deploy.
pub fn build_semantic_state(
    config: &Config,
    vector_store: Arc<dyn CompletionVectorStore>,
) -> Option<SemanticState> {
    if !config.semantic_enabled {
        return None;
    }
    let embedder: Arc<dyn TextEmbed> = match TextEmbedder::load() {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("fatal: ZEROCACHE_SEMANTIC is enabled but the embedder failed to load: {e}");
            std::process::exit(1);
        }
    };
    Some(SemanticState {
        embedder,
        index: Arc::new(SemanticIndex::new()),
        vector_store,
        threshold: config.semantic_threshold,
        match_unit: config.semantic_match_unit,
    })
}

/// Replay persisted vector records into the in-memory index at startup.
pub fn rebuild_index(sem: &SemanticState) {
    let start = std::time::Instant::now();
    let records = match sem.vector_store.load_all() {
        Ok(r) => r,
        Err(e) => {
            warn!("semantic: load_all failed at startup, index starts empty: {e}");
            return;
        }
    };
    let (mut loaded, mut skipped) = (0usize, 0usize);
    for r in records {
        if r.index_version != SEMANTIC_INDEX_VERSION {
            skipped += 1;
            continue;
        }
        sem.index
            .insert(r.scope_hash, &r.vector, r.coarse_key_hash, r.exact_key);
        loaded += 1;
    }
    tracing::info!(
        "semantic: index rebuilt — {loaded} vectors loaded, {skipped} skipped, in {:?}",
        start.elapsed()
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use serde_json::json;
    use zerocache_core::{CacheKey, MatchUnit};
    use zerocache_ports::{CompletionStore, StoreError, VectorRecord};

    use super::*;

    struct ConstEmbedder;
    impl zerocache_semantic::TextEmbed for ConstEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, zerocache_semantic::SemanticError> {
            let mut v = vec![0f32; zerocache_semantic::EMBEDDING_DIM];
            v[0] = 1.0;
            Ok(v)
        }
    }

    struct MemVectorStore(Mutex<Vec<VectorRecord>>);
    impl CompletionVectorStore for MemVectorStore {
        fn insert(&self, r: VectorRecord) -> Result<(), StoreError> {
            self.0.lock().unwrap().push(r);
            Ok(())
        }
        fn delete(&self, k: &CacheKey) -> Result<(), StoreError> {
            self.0.lock().unwrap().retain(|r| &r.exact_key != k);
            Ok(())
        }
        fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError> {
            Ok(self.0.lock().unwrap().clone())
        }
    }

    struct MemCompletionStore(Mutex<HashMap<CacheKey, Vec<u8>>>);
    impl CompletionStore for MemCompletionStore {
        fn get(&self, k: &CacheKey) -> Result<Option<Vec<u8>>, StoreError> {
            Ok(self.0.lock().unwrap().get(k).cloned())
        }
        fn put(&self, k: CacheKey, v: Vec<u8>) -> Result<(), StoreError> {
            self.0.lock().unwrap().insert(k, v);
            Ok(())
        }
        fn delete(&self, k: &CacheKey) -> Result<(), StoreError> {
            self.0.lock().unwrap().remove(k);
            Ok(())
        }
    }

    fn sem_state() -> SemanticState {
        SemanticState {
            embedder: Arc::new(ConstEmbedder),
            index: Arc::new(SemanticIndex::new()),
            vector_store: Arc::new(MemVectorStore(Mutex::new(Vec::new()))),
            threshold: 0.97,
            match_unit: MatchUnit::LastUser,
        }
    }

    fn unit_vec() -> Vec<f32> {
        let mut v = vec![0f32; zerocache_semantic::EMBEDDING_DIM];
        v[0] = 1.0;
        v
    }

    fn body(user: &str) -> serde_json::Value {
        json!({"model":"gpt-4o","messages":[
            {"role":"system","content":"bot"},
            {"role":"user","content": user}
        ],"temperature":0})
    }

    #[test]
    fn semantic_inputs_is_none_without_a_user_message() {
        let b = json!({"model":"x","messages":[{"role":"system","content":"s"}]});
        assert!(semantic_inputs(MatchUnit::LastUser, &b).is_none());
    }

    #[test]
    fn semantic_inputs_coarse_hash_is_stable_across_paraphrases() {
        let (_t1, h1) = semantic_inputs(MatchUnit::LastUser, &body("reset my password")).unwrap();
        let (_t2, h2) =
            semantic_inputs(MatchUnit::LastUser, &body("how do i reset the password")).unwrap();
        assert_eq!(h1, h2);
    }

    #[tokio::test]
    async fn record_then_probe_returns_the_stored_bytes() {
        let sem = sem_state();
        let cs: Arc<dyn CompletionStore> = Arc::new(MemCompletionStore(Mutex::new(HashMap::new())));
        let scope = scope_key(&[1u8; 32], "openai", "scope", "gpt-4o");
        let coarse = [5u8; 32];
        let exact = CacheKey::from_bytes([42u8; 32]);
        cs.put(exact, b"STORED".to_vec()).unwrap();

        record_vector(&sem, exact, scope, coarse, unit_vec()).await;
        let hit = semantic_probe(&sem, &cs, scope, coarse, &unit_vec())
            .await
            .unwrap()
            .expect("semantic hit");
        assert_eq!(hit.record_bytes, b"STORED");
        assert!(hit.score > 0.99);
    }

    #[tokio::test]
    async fn probe_self_heals_when_the_target_blob_is_gone() {
        let sem = sem_state();
        let cs: Arc<dyn CompletionStore> = Arc::new(MemCompletionStore(Mutex::new(HashMap::new())));
        let scope = scope_key(&[1u8; 32], "openai", "scope", "gpt-4o");
        let coarse = [5u8; 32];
        let exact = CacheKey::from_bytes([42u8; 32]);
        record_vector(&sem, exact, scope, coarse, unit_vec()).await;

        assert!(semantic_probe(&sem, &cs, scope, coarse, &unit_vec())
            .await
            .unwrap()
            .is_none());
        assert!(semantic_probe(&sem, &cs, scope, coarse, &unit_vec())
            .await
            .unwrap()
            .is_none());
        assert!(sem.vector_store.load_all().unwrap().is_empty());
    }

    #[tokio::test]
    async fn probe_misses_on_a_wrong_coarse_hash() {
        let sem = sem_state();
        let cs: Arc<dyn CompletionStore> = Arc::new(MemCompletionStore(Mutex::new(HashMap::new())));
        let scope = scope_key(&[1u8; 32], "openai", "scope", "gpt-4o");
        let exact = CacheKey::from_bytes([42u8; 32]);
        cs.put(exact, b"X".to_vec()).unwrap();
        record_vector(&sem, exact, scope, [1u8; 32], unit_vec()).await;
        assert!(semantic_probe(&sem, &cs, scope, [2u8; 32], &unit_vec())
            .await
            .unwrap()
            .is_none());
    }

    #[test]
    fn build_semantic_state_is_none_when_disabled() {
        let mut cfg = crate::config::Config::from_env();
        cfg.semantic_enabled = false;
        let vs: Arc<dyn CompletionVectorStore> = Arc::new(MemVectorStore(Mutex::new(Vec::new())));
        assert!(build_semantic_state(&cfg, vs).is_none());
    }
}

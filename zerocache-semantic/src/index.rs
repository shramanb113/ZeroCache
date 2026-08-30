use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use hnsw_rs::prelude::{DistCosine, Hnsw};

use zerocache_core::CacheKey;

/// blake3 of (owner_id, provider, cache_scope, model). One isolated index per tuple.
pub type ScopeKey = [u8; 32];

pub struct SearchHit {
    pub exact_key: CacheKey,
    pub score: f32,
}

#[derive(Clone)]
struct NodeMeta {
    coarse_key_hash: [u8; 32],
    exact_key: CacheKey,
}

const HNSW_MAX_CONN: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_MAX_LAYER: usize = 16;
const HNSW_CAPACITY_HINT: usize = 16_384;
const SEARCH_K: usize = 8;
const SEARCH_EF: usize = 32;

struct ScopeIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    meta: Vec<NodeMeta>,
    tombstones: HashSet<usize>,
    /// exact_key -> node id. One node per key per scope: `insert` is
    /// idempotent, so this stays 1:1 and makes `tombstone` O(1).
    by_key: HashMap<CacheKey, usize>,
}

impl ScopeIndex {
    fn new() -> Self {
        Self {
            hnsw: Hnsw::new(
                HNSW_MAX_CONN,
                HNSW_CAPACITY_HINT,
                HNSW_MAX_LAYER,
                HNSW_EF_CONSTRUCTION,
                DistCosine {},
            ),
            meta: Vec::new(),
            tombstones: HashSet::new(),
            by_key: HashMap::new(),
        }
    }
}

/// Per-scope HNSW graphs behind one lock: write for insert/tombstone, read for search.
pub struct SemanticIndex {
    inner: RwLock<HashMap<ScopeKey, ScopeIndex>>,
}

impl Default for SemanticIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticIndex {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(
        &self,
        scope: ScopeKey,
        vector: &[f32],
        coarse_key_hash: [u8; 32],
        exact_key: CacheKey,
    ) {
        let mut map = self.inner.write().expect("semantic index lock poisoned");
        let si = map.entry(scope).or_insert_with(ScopeIndex::new);
        if let Some(&id) = si.by_key.get(&exact_key) {
            // Re-delivery (cursor-gap replay, or a writer seeing its own
            // vector back via its poll). The vector is content-addressed by
            // exact_key, so it is byte-identical -- just clear any tombstone.
            si.tombstones.remove(&id);
            return;
        }
        let id = si.meta.len();
        si.meta.push(NodeMeta {
            coarse_key_hash,
            exact_key,
        });
        si.by_key.insert(exact_key, id);
        si.hnsw.insert((vector, id));
    }

    /// Nearest neighbour that also matches `coarse_key_hash` exactly and clears `threshold`.
    pub fn search(
        &self,
        scope: ScopeKey,
        query: &[f32],
        coarse_key_hash: [u8; 32],
        threshold: f32,
    ) -> Option<SearchHit> {
        let map = self.inner.read().expect("semantic index lock poisoned");
        let si = map.get(&scope)?;
        for n in si.hnsw.search(query, SEARCH_K, SEARCH_EF) {
            if si.tombstones.contains(&n.d_id) {
                continue;
            }
            let meta = &si.meta[n.d_id];
            if meta.coarse_key_hash != coarse_key_hash {
                continue;
            }
            let score = 1.0 - n.distance; // DistCosine returns 1 - cosine
            if score >= threshold {
                return Some(SearchHit {
                    exact_key: meta.exact_key,
                    score,
                });
            }
        }
        None
    }

    pub fn tombstone(&self, scope: ScopeKey, exact_key: &CacheKey) {
        let mut map = self.inner.write().expect("semantic index lock poisoned");
        if let Some(si) = map.get_mut(&scope) {
            if let Some(&id) = si.by_key.get(exact_key) {
                si.tombstones.insert(id);
            }
        }
    }

    pub fn len(&self, scope: ScopeKey) -> usize {
        let map = self.inner.read().expect("semantic index lock poisoned");
        map.get(&scope).map(|si| si.meta.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EMBEDDING_DIM;
    use zerocache_core::CacheKey;

    fn key(n: u8) -> CacheKey {
        CacheKey::from_bytes([n; 32])
    }

    fn vec_at(angle: f32) -> Vec<f32> {
        let mut v = vec![0f32; EMBEDDING_DIM];
        v[0] = angle.cos();
        v[1] = angle.sin();
        v
    }

    const SCOPE_A: ScopeKey = [1u8; 32];
    const SCOPE_B: ScopeKey = [2u8; 32];
    const COARSE: [u8; 32] = [3u8; 32];

    #[test]
    fn reinserting_a_live_key_does_not_grow_the_graph() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        assert_eq!(idx.len(SCOPE_A), 1);
        let hit = idx.search(SCOPE_A, &v, COARSE, 0.97).expect("still searchable");
        assert_eq!(hit.exact_key, key(10));
    }

    #[test]
    fn reinserting_a_tombstoned_key_revives_it() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        idx.tombstone(SCOPE_A, &key(10));
        assert!(idx.search(SCOPE_A, &v, COARSE, 0.97).is_none());
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        assert!(
            idx.search(SCOPE_A, &v, COARSE, 0.97).is_some(),
            "a re-delivered vector must clear the tombstone"
        );
        assert_eq!(idx.len(SCOPE_A), 1);
    }

    #[test]
    fn search_returns_the_inserted_node_for_an_identical_vector() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        let hit = idx.search(SCOPE_A, &v, COARSE, 0.97).expect("should hit");
        assert_eq!(hit.exact_key, key(10));
        assert!(hit.score > 0.99, "score {}", hit.score);
    }

    #[test]
    fn search_rejects_a_below_threshold_vector() {
        let idx = SemanticIndex::new();
        idx.insert(SCOPE_A, &vec_at(0.0), COARSE, key(10));
        assert!(idx.search(SCOPE_A, &vec_at(0.93), COARSE, 0.97).is_none());
    }

    #[test]
    fn search_rejects_a_wrong_coarse_key_hash_even_at_cosine_one() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        assert!(idx.search(SCOPE_A, &v, [99u8; 32], 0.97).is_none());
    }

    #[test]
    fn scopes_are_isolated() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        assert!(idx.search(SCOPE_B, &v, COARSE, 0.97).is_none());
    }

    #[test]
    fn a_tombstoned_node_is_never_returned() {
        let idx = SemanticIndex::new();
        let v = vec_at(0.3);
        idx.insert(SCOPE_A, &v, COARSE, key(10));
        idx.tombstone(SCOPE_A, &key(10));
        assert!(idx.search(SCOPE_A, &v, COARSE, 0.97).is_none());
    }

    #[test]
    fn len_counts_inserted_nodes_per_scope() {
        let idx = SemanticIndex::new();
        idx.insert(SCOPE_A, &vec_at(0.1), COARSE, key(1));
        idx.insert(SCOPE_A, &vec_at(0.2), COARSE, key(2));
        assert_eq!(idx.len(SCOPE_A), 2);
        assert_eq!(idx.len(SCOPE_B), 0);
    }
}

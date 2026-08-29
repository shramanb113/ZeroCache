use std::time::{Duration, SystemTime};

use zerocache_core::CacheKey;
use zerocache_ports::{
    CompletionStore, CompletionVectorStore, EmbeddingStore, StoreError, VectorChanges, VectorRecord,
};

pub struct SledStore {
    db: sled::Db,
    // Cached chat-completion response blobs live in their own tree, not the
    // default keyspace: the value encoding differs (an opaque serialized
    // record, not an fp32 vector) and keeping them separate makes each side
    // independently iterable/clearable. `CacheKey::derive_completion` is
    // already domain-separated from `derive`, so this is tidiness, not a
    // correctness requirement.
    completions: sled::Tree,
    // Semantic-index vector records, iterable on their own to rebuild the
    // in-memory index at startup.
    completion_vectors: sled::Tree,
    ttl: Option<Duration>,
}

impl SledStore {
    pub fn open(path: impl AsRef<std::path::Path>, ttl: Option<Duration>) -> sled::Result<Self> {
        let db = sled::open(path)?;
        let completions = db.open_tree("completions")?;
        let completion_vectors = db.open_tree("completion_vectors")?;
        Ok(Self {
            db,
            completions,
            completion_vectors,
            ttl,
        })
    }
}

impl EmbeddingStore for SledStore {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
        let raw = self
            .db
            .get(key.as_bytes())
            .map_err(|e| StoreError(e.to_string()))?;

        let Some(bytes) = raw else { return Ok(None) };
        let (expires_at, vector) = decode(&bytes);

        if let Some(expires_at) = expires_at {
            if SystemTime::now() > expires_at {
                // Lazily remove the expired entry; a failure to remove it
                // isn't fatal to this read, it just means it'll be checked
                // (and removed) again next time it's read.
                let _ = self.db.remove(key.as_bytes());
                return Ok(None);
            }
        }

        Ok(Some(vector))
    }

    fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError> {
        let expires_at = self.ttl.map(|ttl| SystemTime::now() + ttl);
        self.db
            .insert(key.as_bytes(), encode(expires_at, &vector))
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }

    fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
        self.db
            .remove(key.as_bytes())
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }
}

impl CompletionStore for SledStore {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<u8>>, StoreError> {
        let raw = self
            .completions
            .get(key.as_bytes())
            .map_err(|e| StoreError(e.to_string()))?;

        let Some(bytes) = raw else { return Ok(None) };
        let (expires_at, blob) = decode_blob(&bytes);

        if let Some(expires_at) = expires_at {
            if SystemTime::now() > expires_at {
                let _ = self.completions.remove(key.as_bytes());
                return Ok(None);
            }
        }

        Ok(Some(blob))
    }

    fn put(&self, key: CacheKey, value: Vec<u8>) -> Result<(), StoreError> {
        let expires_at = self.ttl.map(|ttl| SystemTime::now() + ttl);
        self.completions
            .insert(key.as_bytes(), encode_blob(expires_at, &value))
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }

    fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
        self.completions
            .remove(key.as_bytes())
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }
}

impl CompletionVectorStore for SledStore {
    fn insert(&self, record: VectorRecord) -> Result<(), StoreError> {
        let expires_at = self.ttl.map(|ttl| SystemTime::now() + ttl);
        self.completion_vectors
            .insert(
                record.exact_key.as_bytes(),
                encode_vector(expires_at, &record),
            )
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }

    fn delete(&self, exact_key: &CacheKey, _scope_hash: &[u8; 32]) -> Result<(), StoreError> {
        self.completion_vectors
            .remove(exact_key.as_bytes())
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }

    // Single process: the in-memory index already sees every write directly
    // through zerocache-http's own index.insert. Nothing to propagate.
    fn changes_since(&self, _cursor: Option<String>) -> Result<VectorChanges, StoreError> {
        Ok(VectorChanges::default())
    }

    fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError> {
        let now = SystemTime::now();
        let mut out = Vec::new();
        for item in self.completion_vectors.iter() {
            let (k, v) = item.map_err(|e| StoreError(e.to_string()))?;
            let key_bytes: [u8; 32] = k
                .as_ref()
                .try_into()
                .map_err(|_| StoreError("vector key is not 32 bytes".into()))?;
            let Some((expires_at, scope_hash, coarse_key_hash, index_version, vector)) =
                decode_vector(&v)
            else {
                continue; // skip a malformed row, don't fail the rebuild
            };
            if let Some(expires_at) = expires_at {
                if now > expires_at {
                    let _ = self.completion_vectors.remove(&k);
                    continue;
                }
            }
            out.push(VectorRecord {
                exact_key: CacheKey::from_bytes(key_bytes),
                scope_hash,
                coarse_key_hash,
                index_version,
                vector,
            });
        }
        Ok(out)
    }
}

fn encode_vector(expires_at: Option<SystemTime>, record: &VectorRecord) -> Vec<u8> {
    let expires_at_secs: u64 = expires_at
        .map(|t| {
            t.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
        .unwrap_or(0);
    let mut out = Vec::with_capacity(8 + 32 + 32 + 1 + record.vector.len() * 4);
    out.extend_from_slice(&expires_at_secs.to_le_bytes());
    out.extend_from_slice(&record.scope_hash);
    out.extend_from_slice(&record.coarse_key_hash);
    out.push(record.index_version);
    out.extend(record.vector.iter().flat_map(|f| f.to_le_bytes()));
    out
}

#[allow(clippy::type_complexity)]
fn decode_vector(bytes: &[u8]) -> Option<(Option<SystemTime>, [u8; 32], [u8; 32], u8, Vec<f32>)> {
    if bytes.len() < 8 + 32 + 32 + 1 {
        return None;
    }
    let expires_at_secs = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    let expires_at = if expires_at_secs == 0 {
        None
    } else {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(expires_at_secs))
    };
    let scope_hash: [u8; 32] = bytes[8..40].try_into().ok()?;
    let coarse_key_hash: [u8; 32] = bytes[40..72].try_into().ok()?;
    let index_version = bytes[72];
    let vector = bytes[73..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Some((
        expires_at,
        scope_hash,
        coarse_key_hash,
        index_version,
        vector,
    ))
}

// Stored value format: [8 bytes: expires_at as unix seconds, LE, 0 = never][fp32 vector bytes].
// A real unix timestamp of exactly 0 (1970-01-01) is never a realistic future
// expiry, so 0 doubles safely as the "no expiry" sentinel — no separate tag
// byte needed.
fn encode(expires_at: Option<SystemTime>, vector: &[f32]) -> Vec<u8> {
    let expires_at_secs: u64 = expires_at
        .map(|t| {
            t.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
        .unwrap_or(0);

    let mut out = Vec::with_capacity(8 + vector.len() * 4);
    out.extend_from_slice(&expires_at_secs.to_le_bytes());
    out.extend(vector.iter().flat_map(|f| f.to_le_bytes()));
    out
}

fn decode(bytes: &[u8]) -> (Option<SystemTime>, Vec<f32>) {
    let expires_at_secs = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .expect("stored value always has an 8-byte expiry prefix"),
    );
    let expires_at = if expires_at_secs == 0 {
        None
    } else {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(expires_at_secs))
    };

    let vector = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    (expires_at, vector)
}

// Same 8-byte little-endian expiry prefix as `encode`/`decode` above (0 =
// never expires), but the payload is an opaque blob (a serialized
// completion record) rather than an fp32 vector.
fn encode_blob(expires_at: Option<SystemTime>, blob: &[u8]) -> Vec<u8> {
    let expires_at_secs: u64 = expires_at
        .map(|t| {
            t.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        })
        .unwrap_or(0);

    let mut out = Vec::with_capacity(8 + blob.len());
    out.extend_from_slice(&expires_at_secs.to_le_bytes());
    out.extend_from_slice(blob);
    out
}

fn decode_blob(bytes: &[u8]) -> (Option<SystemTime>, Vec<u8>) {
    let expires_at_secs = u64::from_le_bytes(
        bytes[0..8]
            .try_into()
            .expect("stored value always has an 8-byte expiry prefix"),
    );
    let expires_at = if expires_at_secs == 0 {
        None
    } else {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(expires_at_secs))
    };

    (expires_at, bytes[8..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "zerocache-sled-test-{}-{}",
            std::process::id(),
            rand_suffix()
        ))
    }

    // No external RNG dependency for a unique-enough test directory name —
    // nanosecond timestamp is sufficient to avoid collisions between the
    // tests in this file, which is all that's needed here.
    fn rand_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    #[test]
    fn put_then_get_roundtrips() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "test-scope", "m", "v1", "hello");

        assert_eq!(EmbeddingStore::get(&store, &key).unwrap(), None);
        EmbeddingStore::put(&store, key, vec![1.0, 2.5, -3.25]).unwrap();
        assert_eq!(
            EmbeddingStore::get(&store, &key).unwrap(),
            Some(vec![1.0, 2.5, -3.25])
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_removes_an_entry() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive(
            [1u8; 32],
            "openai",
            "test-scope",
            "m",
            "v1",
            "to be deleted",
        );

        EmbeddingStore::put(&store, key, vec![1.0]).unwrap();
        assert_eq!(EmbeddingStore::get(&store, &key).unwrap(), Some(vec![1.0]));

        EmbeddingStore::delete(&store, &key).unwrap();
        assert_eq!(EmbeddingStore::get(&store, &key).unwrap(), None);

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_on_a_missing_key_is_not_an_error() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive(
            [1u8; 32],
            "openai",
            "test-scope",
            "m",
            "v1",
            "never existed",
        );

        assert!(EmbeddingStore::delete(&store, &key).is_ok());

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn entry_past_its_ttl_reads_as_a_miss() {
        let dir = temp_dir();
        // A TTL of zero duration means "already expired" the moment it's written.
        let store = SledStore::open(&dir, Some(Duration::from_secs(0))).unwrap();
        let key = CacheKey::derive(
            [1u8; 32],
            "openai",
            "test-scope",
            "m",
            "v1",
            "expires immediately",
        );

        EmbeddingStore::put(&store, key, vec![1.0]).unwrap();
        // The clock must advance past the expiry instant, which a zero-second
        // TTL guarantees for any nonzero wall-clock delay between put and get.
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(
            EmbeddingStore::get(&store, &key).unwrap(),
            None,
            "an expired entry must read as a miss, not a hit"
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn entry_within_its_ttl_still_reads_as_a_hit() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, Some(Duration::from_secs(3600))).unwrap();
        let key = CacheKey::derive(
            [1u8; 32],
            "openai",
            "test-scope",
            "m",
            "v1",
            "expires in an hour",
        );

        EmbeddingStore::put(&store, key, vec![1.0]).unwrap();
        assert_eq!(
            EmbeddingStore::get(&store, &key).unwrap(),
            Some(vec![1.0]),
            "an entry well within its TTL must still hit"
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn no_ttl_configured_means_entries_never_expire() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive(
            [1u8; 32],
            "openai",
            "test-scope",
            "m",
            "v1",
            "lives forever",
        );

        EmbeddingStore::put(&store, key, vec![1.0]).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(
            EmbeddingStore::get(&store, &key).unwrap(),
            Some(vec![1.0]),
            "with no TTL configured, an entry must never expire"
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    fn completion_key(text: &str) -> CacheKey {
        CacheKey::derive_completion([1u8; 32], "openai", "test-scope", "gpt-4o", "v1", text)
    }

    #[test]
    fn completion_put_then_get_roundtrips() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = completion_key("req-a");

        assert_eq!(CompletionStore::get(&store, &key).unwrap(), None);
        CompletionStore::put(&store, key, b"{\"body\":1}".to_vec()).unwrap();
        assert_eq!(
            CompletionStore::get(&store, &key).unwrap(),
            Some(b"{\"body\":1}".to_vec())
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn completion_delete_removes_an_entry() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = completion_key("req-b");

        CompletionStore::put(&store, key, b"blob".to_vec()).unwrap();
        CompletionStore::delete(&store, &key).unwrap();
        assert_eq!(CompletionStore::get(&store, &key).unwrap(), None);

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn completion_delete_on_a_missing_key_is_not_an_error() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        assert!(CompletionStore::delete(&store, &completion_key("never")).is_ok());

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn completion_entry_past_its_ttl_reads_as_a_miss() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, Some(Duration::from_secs(0))).unwrap();
        let key = completion_key("expires-now");

        CompletionStore::put(&store, key, b"blob".to_vec()).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(
            CompletionStore::get(&store, &key).unwrap(),
            None,
            "an expired completion entry must read as a miss"
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn completion_entry_within_its_ttl_still_reads_as_a_hit() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, Some(Duration::from_secs(3600))).unwrap();
        let key = completion_key("expires-in-an-hour");

        CompletionStore::put(&store, key, b"blob".to_vec()).unwrap();
        assert_eq!(
            CompletionStore::get(&store, &key).unwrap(),
            Some(b"blob".to_vec())
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn completion_and_embedding_entries_do_not_interfere() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let embedding_key =
            CacheKey::derive([1u8; 32], "openai", "test-scope", "gpt-4o", "v1", "shared");
        let comp_key = completion_key("shared");

        EmbeddingStore::put(&store, embedding_key, vec![1.0, 2.0]).unwrap();
        CompletionStore::put(&store, comp_key, b"completion-blob".to_vec()).unwrap();

        assert_eq!(
            EmbeddingStore::get(&store, &embedding_key).unwrap(),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(
            CompletionStore::get(&store, &comp_key).unwrap(),
            Some(b"completion-blob".to_vec())
        );

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    fn sample_record(text: &str, version: u8) -> VectorRecord {
        VectorRecord {
            exact_key: completion_key(text),
            scope_hash: [9u8; 32],
            coarse_key_hash: [4u8; 32],
            index_version: version,
            vector: (0..384).map(|i| i as f32 * 0.01).collect(),
        }
    }

    #[test]
    fn vector_insert_then_load_all_round_trips_every_field() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let rec = sample_record("req-1", 1);
        CompletionVectorStore::insert(&store, rec.clone()).unwrap();

        let loaded = CompletionVectorStore::load_all(&store).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].exact_key, rec.exact_key);
        assert_eq!(loaded[0].scope_hash, rec.scope_hash);
        assert_eq!(loaded[0].coarse_key_hash, rec.coarse_key_hash);
        assert_eq!(loaded[0].index_version, 1);
        assert_eq!(loaded[0].vector, rec.vector);

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn vector_delete_removes_it_from_load_all() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let rec = sample_record("req-2", 1);
        CompletionVectorStore::insert(&store, rec.clone()).unwrap();
        CompletionVectorStore::delete(&store, &rec.exact_key, &rec.scope_hash).unwrap();
        assert!(CompletionVectorStore::load_all(&store).unwrap().is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn changes_since_is_a_noop_on_sled() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let c = CompletionVectorStore::changes_since(&store, None).unwrap();
        assert!(c.upserts.is_empty() && c.deletes.is_empty() && c.cursor.is_none());
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn vector_records_past_their_ttl_are_absent_from_load_all() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, Some(Duration::from_secs(0))).unwrap();
        CompletionVectorStore::insert(&store, sample_record("req-3", 1)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert!(CompletionVectorStore::load_all(&store).unwrap().is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn vector_load_all_returns_mixed_index_versions_verbatim() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        CompletionVectorStore::insert(&store, sample_record("v1", 1)).unwrap();
        CompletionVectorStore::insert(&store, sample_record("v2", 2)).unwrap();
        let mut versions: Vec<u8> = CompletionVectorStore::load_all(&store)
            .unwrap()
            .iter()
            .map(|r| r.index_version)
            .collect();
        versions.sort();
        assert_eq!(versions, vec![1, 2]);
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn vector_tree_does_not_interfere_with_the_completion_or_embedding_trees() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = completion_key("shared");
        EmbeddingStore::put(&store, key, vec![1.0]).unwrap();
        CompletionStore::put(&store, key, b"blob".to_vec()).unwrap();
        CompletionVectorStore::insert(&store, sample_record("shared", 1)).unwrap();

        assert_eq!(EmbeddingStore::get(&store, &key).unwrap(), Some(vec![1.0]));
        assert_eq!(
            CompletionStore::get(&store, &key).unwrap(),
            Some(b"blob".to_vec())
        );
        assert_eq!(CompletionVectorStore::load_all(&store).unwrap().len(), 1);
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }
}

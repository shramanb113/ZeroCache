use std::time::{Duration, SystemTime};

use zerocache_core::CacheKey;
use zerocache_ports::{EmbeddingStore, StoreError};

pub struct SledStore {
    db: sled::Db,
    ttl: Option<Duration>,
}

impl SledStore {
    pub fn open(path: impl AsRef<std::path::Path>, ttl: Option<Duration>) -> sled::Result<Self> {
        Ok(Self { db: sled::open(path)?, ttl })
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

// Stored value format: [8 bytes: expires_at as unix seconds, LE, 0 = never][fp32 vector bytes].
// A real unix timestamp of exactly 0 (1970-01-01) is never a realistic future
// expiry, so 0 doubles safely as the "no expiry" sentinel — no separate tag
// byte needed.
fn encode(expires_at: Option<SystemTime>, vector: &[f32]) -> Vec<u8> {
    let expires_at_secs: u64 = expires_at
        .map(|t| t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs())
        .unwrap_or(0);

    let mut out = Vec::with_capacity(8 + vector.len() * 4);
    out.extend_from_slice(&expires_at_secs.to_le_bytes());
    out.extend(vector.iter().flat_map(|f| f.to_le_bytes()));
    out
}

fn decode(bytes: &[u8]) -> (Option<SystemTime>, Vec<f32>) {
    let expires_at_secs = u64::from_le_bytes(bytes[0..8].try_into().expect("stored value always has an 8-byte expiry prefix"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("zerocache-sled-test-{}-{}", std::process::id(), rand_suffix()))
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
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "hello");

        assert_eq!(store.get(&key).unwrap(), None);
        store.put(key, vec![1.0, 2.5, -3.25]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0, 2.5, -3.25]));

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_removes_an_entry() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "to be deleted");

        store.put(key, vec![1.0]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]));

        store.delete(&key).unwrap();
        assert_eq!(store.get(&key).unwrap(), None);

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_on_a_missing_key_is_not_an_error() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "never existed");

        assert!(store.delete(&key).is_ok());

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn entry_past_its_ttl_reads_as_a_miss() {
        let dir = temp_dir();
        // A TTL of zero duration means "already expired" the moment it's written.
        let store = SledStore::open(&dir, Some(Duration::from_secs(0))).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "expires immediately");

        store.put(key, vec![1.0]).unwrap();
        // The clock must advance past the expiry instant, which a zero-second
        // TTL guarantees for any nonzero wall-clock delay between put and get.
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(store.get(&key).unwrap(), None, "an expired entry must read as a miss, not a hit");

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn entry_within_its_ttl_still_reads_as_a_hit() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, Some(Duration::from_secs(3600))).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "expires in an hour");

        store.put(key, vec![1.0]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]), "an entry well within its TTL must still hit");

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn no_ttl_configured_means_entries_never_expire() {
        let dir = temp_dir();
        let store = SledStore::open(&dir, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "lives forever");

        store.put(key, vec![1.0]).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]), "with no TTL configured, an entry must never expire");

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }
}

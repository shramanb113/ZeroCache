use zerocache_core::CacheKey;
use zerocache_ports::EmbeddingStore;

pub struct SledStore {
    db: sled::Db,
}

impl SledStore {
    pub fn open(path: impl AsRef<std::path::Path>) -> sled::Result<Self> {
        Ok(Self { db: sled::open(path)? })
    }
}

impl EmbeddingStore for SledStore {
    fn get(&self, key: &CacheKey) -> Option<Vec<f32>> {
        let raw = self.db.get(key.as_bytes()).ok().flatten()?;
        Some(decode(&raw))
    }

    fn put(&self, key: CacheKey, vector: Vec<f32>) {
        let _ = self.db.insert(key.as_bytes(), encode(&vector));
    }
}

fn encode(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_roundtrips() {
        let dir = std::env::temp_dir().join(format!("zerocache-sled-test-{}", std::process::id()));
        let store = SledStore::open(&dir).unwrap();
        let key = CacheKey::derive("m", "v1", "hello");

        assert_eq!(store.get(&key), None);
        store.put(key, vec![1.0, 2.5, -3.25]);
        assert_eq!(store.get(&key), Some(vec![1.0, 2.5, -3.25]));

        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }
}

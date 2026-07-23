use std::time::Duration;

use redis::Commands;
use zerocache_core::CacheKey;
use zerocache_ports::{EmbeddingStore, StoreError};

// Pooled, not a single shared connection behind a mutex: concurrent requests
// each check out their own connection, so no lock serializes cache access.
// Safe across multiple pods because the cache key is content-addressed —
// two replicas racing to fill the same key both compute the same vector,
// so a last-write-wins SET is not a correctness problem.
pub struct RedisStore {
    pool: r2d2::Pool<redis::Client>,
    ttl: Option<Duration>,
}

impl RedisStore {
    pub fn connect(redis_url: &str, ttl: Option<Duration>) -> Result<Self, StoreError> {
        let client = redis::Client::open(redis_url).map_err(|e| StoreError(e.to_string()))?;
        let pool = r2d2::Pool::builder()
            .build(client)
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(Self { pool, ttl })
    }
}

impl EmbeddingStore for RedisStore {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        let raw: Option<Vec<u8>> = conn
            .get(redis_key(key))
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(raw.map(|bytes| decode(&bytes)))
    }

    fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        match self.ttl {
            // Redis handles expiry natively -- no stored-value format change
            // needed, unlike sled.
            Some(ttl) => {
                conn.set_ex::<_, _, ()>(redis_key(&key), encode(&vector), ttl.as_secs())
                    .map_err(|e| StoreError(e.to_string()))?;
            }
            None => {
                conn.set::<_, _, ()>(redis_key(&key), encode(&vector))
                    .map_err(|e| StoreError(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        conn.del::<_, ()>(redis_key(key)).map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }
}

fn redis_key(key: &CacheKey) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + 32);
    out.extend_from_slice(b"zerocache:");
    out.extend_from_slice(key.as_bytes());
    out
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

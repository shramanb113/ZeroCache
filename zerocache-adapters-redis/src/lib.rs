use std::time::Duration;

use redis::Commands;
use zerocache_core::CacheKey;
use zerocache_ports::{EmbeddingStore, StoreError};

// redis-rs sets no socket timeout by default, so a stale or half-dead TCP
// connection (the Redis process died without a clean FIN, or a network
// partition silently drops packets) can otherwise block a synchronous call
// forever. 5s is a conservative ceiling for a same-network Redis round-trip,
// not a measured SLA -- mirroring PROVIDER_TIMEOUT's rationale in the
// provider adapters. Applied to every checked-out connection so a stale
// connection surfaces as a fast error instead of hanging the caller (and,
// via zerocache-http's own STORE_TIMEOUT wrapper, blocking that request's
// response and the server's graceful-shutdown drain) indefinitely.
const STORE_TIMEOUT: Duration = Duration::from_secs(5);

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

// Bounds both directions of the socket so neither a stalled read nor a
// stalled write on a stale connection can hang the caller. Applied fresh on
// every checkout rather than once at pool-build time: pooled connections
// are reused across many calls, and re-applying the same value each time is
// a cheap setsockopt, not a real cost.
fn apply_socket_timeouts(conn: &r2d2::PooledConnection<redis::Client>) -> Result<(), StoreError> {
    conn.set_read_timeout(Some(STORE_TIMEOUT)).map_err(|e| StoreError(e.to_string()))?;
    conn.set_write_timeout(Some(STORE_TIMEOUT)).map_err(|e| StoreError(e.to_string()))?;
    Ok(())
}

impl EmbeddingStore for RedisStore {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        apply_socket_timeouts(&conn)?;
        let raw: Option<Vec<u8>> = conn
            .get(redis_key(key))
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(raw.map(|bytes| decode(&bytes)))
    }

    fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError> {
        let mut conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        apply_socket_timeouts(&conn)?;
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
        apply_socket_timeouts(&conn)?;
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

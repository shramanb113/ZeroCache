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

// Integration tests against a real, ephemeral Redis instance via
// testcontainers -- not mocks, not a manually-started local Redis. Ignored
// by default so the workspace's documented `cargo test --workspace` command
// stays fast and requires no external service; run explicitly with:
//   cargo test -p zerocache-adapters-redis -- --ignored
// Requires Docker running locally or in CI. Each test starts its own
// container rather than sharing one -- slower, but keeps every test fully
// independent with no shared Redis state to reset between runs.
#[cfg(test)]
mod live_redis_tests {
    use testcontainers_modules::{
        redis::{Redis, REDIS_PORT},
        testcontainers::{runners::SyncRunner, Container},
    };

    use super::*;

    fn start_redis() -> (Container<Redis>, String) {
        let container = Redis::default().start().expect("failed to start Redis testcontainer -- is Docker running?");
        let host = container.get_host().expect("failed to get testcontainer host");
        let port = container.get_host_port_ipv4(REDIS_PORT).expect("failed to get testcontainer port");
        (container, format!("redis://{host}:{port}"))
    }

    #[test]
    #[ignore]
    fn put_then_get_roundtrips_against_a_real_redis() {
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "hello");

        assert_eq!(store.get(&key).unwrap(), None);
        store.put(key, vec![1.0, 2.5, -3.25]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0, 2.5, -3.25]));
    }

    #[test]
    #[ignore]
    fn delete_removes_an_entry_from_a_real_redis() {
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "to be deleted");

        store.put(key, vec![1.0]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]));

        store.delete(&key).unwrap();
        assert_eq!(store.get(&key).unwrap(), None);
    }

    #[test]
    #[ignore]
    fn delete_on_a_missing_key_is_not_an_error_on_a_real_redis() {
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "never existed");

        assert!(store.delete(&key).is_ok());
    }

    #[test]
    #[ignore]
    fn entry_past_its_ttl_reads_as_a_miss_on_a_real_redis() {
        let (_container, url) = start_redis();
        // A literal 0-second TTL is rejected by Redis's SET...EX (which is
        // exactly why zerocache-http's config layer guards against it before
        // it ever reaches this adapter -- see ZEROCACHE_TTL_SECONDS=0
        // handling). Use the smallest real TTL Redis accepts and sleep past
        // it, rather than exercising that unrelated edge case here.
        let store = RedisStore::connect(&url, Some(Duration::from_secs(1))).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "expires almost immediately");

        store.put(key, vec![1.0]).unwrap();
        std::thread::sleep(Duration::from_millis(1500));
        assert_eq!(store.get(&key).unwrap(), None, "an entry past its Redis-native TTL must read as a miss");
    }

    #[test]
    #[ignore]
    fn entry_within_its_ttl_still_reads_as_a_hit_on_a_real_redis() {
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, Some(Duration::from_secs(3600))).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "expires in an hour");

        store.put(key, vec![1.0]).unwrap();
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]), "an entry well within its TTL must still hit");
    }

    #[test]
    #[ignore]
    fn no_ttl_configured_means_entries_never_expire_on_a_real_redis() {
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();
        let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", "lives forever");

        store.put(key, vec![1.0]).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(store.get(&key).unwrap(), Some(vec![1.0]), "with no TTL configured, an entry must never expire");
    }

    #[test]
    #[ignore]
    fn socket_timeouts_do_not_break_normal_fast_operations_against_a_real_redis() {
        // Sanity check that apply_socket_timeouts (a 5s read/write timeout
        // applied on every checkout) doesn't interfere with ordinary, fast
        // round-trips against a real connection -- a regression here would
        // mean every real request starts silently erroring.
        let (_container, url) = start_redis();
        let store = RedisStore::connect(&url, None).unwrap();

        for i in 0..20 {
            let key = CacheKey::derive([1u8; 32], "openai", "m", "v1", &format!("fast op {i}"));
            store.put(key, vec![i as f32]).unwrap();
            assert_eq!(store.get(&key).unwrap(), Some(vec![i as f32]));
        }
    }
}

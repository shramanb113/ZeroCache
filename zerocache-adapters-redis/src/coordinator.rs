use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use zerocache_core::CacheKey;
use zerocache_ports::{CoalescingCoordinator, FollowSignal, Role, StoreError};

const LOCK_TTL: Duration = Duration::from_secs(60);
const LOCK_PREFIX: &[u8] = b"zerocache:coalesce:lock:";
const DONE_PREFIX: &[u8] = b"zerocache:coalesce:done:";
const SOCKET_TIMEOUT: Duration = Duration::from_secs(5);
// Coarse per-coordinator log rate limit so a Redis outage can't flood logs.
const WARN_INTERVAL_MS: u64 = 10_000;

/// Redis-backed distributed single-flight. A `SET NX PX` lock elects one
/// leader per key; a `PUBLISH` on release wakes followers, backed up by the
/// caller's own periodic store re-read (see crate::coalesce). All Redis I/O
/// is synchronous; the caller runs this on a blocking thread.
pub struct RedisCoordinator {
    pool: r2d2::Pool<redis::Client>,
    // used by the Task 9 pub/sub reader thread
    #[expect(dead_code)]
    client: redis::Client,
    replica_id: Vec<u8>,
    lock_ttl: Duration,
    last_warn_ms: AtomicU64,
    // Task 9 adds: slots: Arc<Mutex<HashMap<CacheKey, Arc<KeySlot>>>>,
}

fn lock_key(key: &CacheKey) -> Vec<u8> {
    [LOCK_PREFIX, key.as_bytes()].concat()
}

fn done_channel(key: &CacheKey) -> Vec<u8> {
    [DONE_PREFIX, key.as_bytes()].concat()
}

fn new_replica_id() -> Vec<u8> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos).into_bytes()
}

impl RedisCoordinator {
    pub fn connect(redis_url: &str) -> Result<Self, StoreError> {
        Self::build(redis_url, LOCK_TTL)
    }

    #[cfg(test)]
    pub(crate) fn connect_with_lock_ttl(
        redis_url: &str,
        ttl: Duration,
    ) -> Result<Self, StoreError> {
        Self::build(redis_url, ttl)
    }

    fn build(redis_url: &str, lock_ttl: Duration) -> Result<Self, StoreError> {
        let client = redis::Client::open(redis_url).map_err(|e| StoreError(e.to_string()))?;
        let pool = r2d2::Pool::builder()
            .max_size(4)
            .build(client.clone())
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(Self {
            pool,
            client,
            replica_id: new_replica_id(),
            lock_ttl,
            last_warn_ms: AtomicU64::new(0),
        })
    }

    fn conn(&self) -> Result<r2d2::PooledConnection<redis::Client>, StoreError> {
        let conn = self.pool.get().map_err(|e| StoreError(e.to_string()))?;
        conn.set_read_timeout(Some(SOCKET_TIMEOUT))
            .map_err(|e| StoreError(e.to_string()))?;
        conn.set_write_timeout(Some(SOCKET_TIMEOUT))
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(conn)
    }

    /// Logs at `warn`, but at most once per `WARN_INTERVAL_MS` per coordinator
    /// so a sustained Redis outage doesn't flood the log. A small race between
    /// two near-simultaneous callers is acceptable -- this is coarse by design.
    fn warn_throttled(&self, msg: std::fmt::Arguments<'_>) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = self.last_warn_ms.load(Ordering::Relaxed);
        if now.saturating_sub(last) >= WARN_INTERVAL_MS {
            self.last_warn_ms.store(now, Ordering::Relaxed);
            tracing::warn!("{msg}");
        }
    }

    fn try_lead_inner(&self, key: &CacheKey) -> Result<Role, StoreError> {
        let mut conn = self.conn()?;
        let set: Option<String> = redis::cmd("SET")
            .arg(lock_key(key))
            .arg(&self.replica_id)
            .arg("NX")
            .arg("PX")
            .arg(self.lock_ttl.as_millis() as u64)
            .query(&mut *conn)
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(if set.is_some() {
            Role::Leader
        } else {
            Role::Follower
        })
    }

    fn complete_inner(&self, key: &CacheKey) -> Result<(), StoreError> {
        let mut conn = self.conn()?;
        // check-and-delete: only remove the lock if we still own it
        let script = redis::Script::new(
            "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
        );
        script
            .key(lock_key(key))
            .arg(&self.replica_id)
            .invoke::<i64>(&mut *conn)
            .map_err(|e| StoreError(e.to_string()))?;
        redis::cmd("PUBLISH")
            .arg(done_channel(key))
            .arg(b"1".to_vec())
            .query::<()>(&mut *conn)
            .map_err(|e| StoreError(e.to_string()))?;
        Ok(())
    }
}

impl CoalescingCoordinator for RedisCoordinator {
    fn try_lead(&self, key: &CacheKey) -> Role {
        match self.try_lead_inner(key) {
            Ok(role) => role,
            Err(e) => {
                self.warn_throttled(format_args!(
                    "RedisCoordinator.try_lead failed ({e}) -- leading without coordination"
                ));
                Role::Leader
            }
        }
    }

    fn complete(&self, key: &CacheKey) {
        if let Err(e) = self.complete_inner(key) {
            self.warn_throttled(format_args!(
                "RedisCoordinator.complete failed ({e}) -- lock will expire via TTL"
            ));
        }
    }

    fn follow(&self, _key: &CacheKey, wait: Duration) -> FollowSignal {
        // Task 9 replaces this with a real pub/sub-backed wait.
        std::thread::sleep(wait);
        FollowSignal::WaitElapsed
    }
}

// Integration tests against a real, ephemeral Redis via testcontainers.
// Ignored by default so `cargo test --workspace` needs no Docker; run with:
//   cargo test -p zerocache-adapters-redis --lib coordinator -- --ignored
#[cfg(test)]
mod live_coordinator_tests {
    use std::time::Duration;

    use testcontainers_modules::{
        redis::{Redis, REDIS_PORT},
        testcontainers::{runners::SyncRunner, Container},
    };
    use zerocache_core::CacheKey;
    use zerocache_ports::{CoalescingCoordinator, Role};

    use super::*;

    fn start_redis() -> (Container<Redis>, String) {
        let c = Redis::default()
            .start()
            .expect("failed to start Redis testcontainer -- is Docker running?");
        let host = c.get_host().expect("host");
        let port = c.get_host_port_ipv4(REDIS_PORT).expect("port");
        (c, format!("redis://{host}:{port}"))
    }

    fn key(n: u8) -> CacheKey {
        CacheKey::from_bytes([n; 32])
    }

    #[test]
    #[ignore]
    fn first_replica_leads_and_a_second_follows_the_same_key() {
        let (_c, url) = start_redis();
        let a = RedisCoordinator::connect(&url).unwrap();
        let b = RedisCoordinator::connect(&url).unwrap();
        let k = key(1);
        assert_eq!(a.try_lead(&k), Role::Leader);
        assert_eq!(b.try_lead(&k), Role::Follower);
    }

    #[test]
    #[ignore]
    fn complete_releases_the_lock_so_a_later_contender_leads() {
        let (_c, url) = start_redis();
        let a = RedisCoordinator::connect(&url).unwrap();
        let b = RedisCoordinator::connect(&url).unwrap();
        let k = key(2);
        assert_eq!(a.try_lead(&k), Role::Leader);
        a.complete(&k);
        assert_eq!(b.try_lead(&k), Role::Leader);
    }

    #[test]
    #[ignore]
    fn an_unreleased_lock_expires_after_its_ttl() {
        let (_c, url) = start_redis();
        let a = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(1)).unwrap();
        let b = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(1)).unwrap();
        let k = key(3);
        assert_eq!(a.try_lead(&k), Role::Leader);
        assert_eq!(b.try_lead(&k), Role::Follower);
        std::thread::sleep(Duration::from_millis(1300));
        assert_eq!(b.try_lead(&k), Role::Leader, "the lock should have expired");
    }

    #[test]
    #[ignore]
    fn complete_does_not_delete_a_lock_owned_by_someone_else() {
        let (_c, url) = start_redis();
        let a = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(1)).unwrap();
        let b = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(1)).unwrap();
        let k = key(4);
        assert_eq!(a.try_lead(&k), Role::Leader);
        std::thread::sleep(Duration::from_millis(1300)); // a's lock expires
        assert_eq!(b.try_lead(&k), Role::Leader); // b now owns it
        a.complete(&k); // must NOT remove b's lock
                        // a fresh contender still finds the lock held by b
        let c = RedisCoordinator::connect(&url).unwrap();
        assert_eq!(c.try_lead(&k), Role::Follower);
    }
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use zerocache_core::CacheKey;
use zerocache_ports::{CoalescingCoordinator, FollowSignal, Role, StoreError};

const LOCK_TTL: Duration = Duration::from_secs(60);
const LOCK_PREFIX: &[u8] = b"zerocache:coalesce:lock:";
const DONE_PREFIX: &[u8] = b"zerocache:coalesce:done:";
const SOCKET_TIMEOUT: Duration = Duration::from_secs(5);
// Coarse per-coordinator log rate limit so a Redis outage can't flood logs.
const WARN_INTERVAL_MS: u64 = 10_000;
// Followers with no `follow` call for this long have their slot swept.
const SLOT_TTL: Duration = Duration::from_secs(90);
// build() waits at most this long for the reader's first psubscribe.
const READER_SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);

/// One waiter rendezvous for a key. The flag flips true when a `done:<key>`
/// message arrives (from any replica, ours included) and stays set until the
/// `SLOT_TTL` sweep drops the slot -- a later `follow` for the same key then
/// returns `Signalled` at once, which is harmless: the caller only re-reads
/// the store and exits on the hit or its own deadline.
struct KeySlot {
    lock: Mutex<bool>,
    cv: Condvar,
    last_used: Mutex<Instant>,
}

impl KeySlot {
    fn new() -> Self {
        Self {
            lock: Mutex::new(false),
            cv: Condvar::new(),
            last_used: Mutex::new(Instant::now()),
        }
    }
}

type SlotMap = Arc<Mutex<HashMap<CacheKey, Arc<KeySlot>>>>;

/// Redis-backed distributed single-flight. A `SET NX PX` lock elects one
/// leader per key; a `PUBLISH` on release wakes followers, backed up by the
/// caller's own periodic store re-read (see the coalescing layer in
/// zerocache-http). All Redis I/O is synchronous; the caller runs this on a
/// blocking thread.
pub struct RedisCoordinator {
    pool: r2d2::Pool<redis::Client>,
    replica_id: Vec<u8>,
    lock_ttl: Duration,
    last_warn_ms: AtomicU64,
    slots: SlotMap,
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

/// Detached thread: psubscribe `done:*` forever, reconnecting on any error.
/// Signals `ready_tx` once, right after the first successful psubscribe.
fn spawn_pubsub_reader(client: redis::Client, slots: SlotMap, ready_tx: mpsc::Sender<()>) {
    std::thread::Builder::new()
        .name("zerocache-coalesce-pubsub".into())
        .spawn(move || {
            // Single-threaded loop; a plain local throttles the reconnect warn.
            let mut last_warn: Option<Instant> = None;
            // Taken and sent once, after the first psubscribe succeeds.
            let mut ready_tx = Some(ready_tx);
            loop {
                match run_pubsub_reader(&client, &slots, &mut ready_tx) {
                    Ok(()) => {} // unreachable: the inner loop only exits via Err
                    Err(e) => {
                        if last_warn
                            .is_none_or(|t| t.elapsed().as_millis() as u64 >= WARN_INTERVAL_MS)
                        {
                            tracing::warn!(
                                "coalesce pub/sub reader dropped ({e}); reconnecting in 1s"
                            );
                            last_warn = Some(Instant::now());
                        }
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            }
        })
        .expect("failed to spawn coalesce pub/sub reader thread");
}

/// Blocks on `done:*` messages, flipping the matching key's slot flag and
/// waking its condvar. Returns only on a Redis error, prompting a reconnect.
fn run_pubsub_reader(
    client: &redis::Client,
    slots: &SlotMap,
    ready_tx: &mut Option<mpsc::Sender<()>>,
) -> Result<(), StoreError> {
    let mut conn = client
        .get_connection()
        .map_err(|e| StoreError(e.to_string()))?;
    let mut pubsub = conn.as_pubsub();
    let mut pattern = DONE_PREFIX.to_vec();
    pattern.push(b'*');
    pubsub
        .psubscribe(pattern)
        .map_err(|e| StoreError(e.to_string()))?;
    // Subscribed: unblock build(). A send after build() gave up returns Err.
    if let Some(tx) = ready_tx.take() {
        let _ = tx.send(());
    }
    loop {
        let msg = pubsub
            .get_message()
            .map_err(|e| StoreError(e.to_string()))?;
        // Channel is DONE_PREFIX + 32 raw key bytes -- not necessarily UTF-8.
        let channel: Vec<u8> = match msg.get_channel() {
            Ok(c) => c,
            Err(_) => continue,
        };
        if channel.len() != DONE_PREFIX.len() + 32 || &channel[..DONE_PREFIX.len()] != DONE_PREFIX {
            continue;
        }
        let mut raw = [0u8; 32];
        raw.copy_from_slice(&channel[DONE_PREFIX.len()..]);
        let key = CacheKey::from_bytes(raw);
        if let Some(slot) = slots.lock().unwrap().get(&key).cloned() {
            *slot.lock.lock().unwrap() = true;
            slot.cv.notify_all();
        }
    }
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
        let slots: SlotMap = Arc::new(Mutex::new(HashMap::new()));
        let (ready_tx, ready_rx) = mpsc::channel();
        // The reader owns its own client clone; nothing else needs a raw one.
        spawn_pubsub_reader(client, Arc::clone(&slots), ready_tx);
        // Block until the reader has psubscribed, so a PUBLISH racing a
        // just-returned connect() is still delivered. Bounded so a Redis
        // outage at startup can't hang boot.
        if ready_rx.recv_timeout(READER_SUBSCRIBE_TIMEOUT).is_err() {
            tracing::warn!(
                "coalesce pub/sub reader not ready after {:?}; proceeding (wakes fall back to the caller's poll re-read)",
                READER_SUBSCRIBE_TIMEOUT
            );
        }
        Ok(Self {
            pool,
            replica_id: new_replica_id(),
            lock_ttl,
            last_warn_ms: AtomicU64::new(0),
            slots,
        })
    }

    /// Get-or-create this key's slot, sweeping entries no `follow` has touched
    /// in `SLOT_TTL`.
    fn slot_for(&self, key: &CacheKey) -> Arc<KeySlot> {
        let mut map = self.slots.lock().unwrap();
        let now = Instant::now();
        map.retain(|_, s| now.duration_since(*s.last_used.lock().unwrap()) < SLOT_TTL);
        let slot = Arc::clone(map.entry(*key).or_insert_with(|| Arc::new(KeySlot::new())));
        *slot.last_used.lock().unwrap() = now;
        slot
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

    fn follow(&self, key: &CacheKey, wait: Duration) -> FollowSignal {
        let slot = self.slot_for(key);
        let guard = slot.lock.lock().unwrap();
        if *guard {
            return FollowSignal::Signalled;
        }
        let (guard, timeout) = slot
            .cv
            .wait_timeout(guard, wait)
            .expect("coalesce slot condvar poisoned");
        // A spurious wakeup that didn't time out is treated as "re-check the
        // store" -- safe, the caller re-reads and exits on hit-or-deadline.
        if *guard || !timeout.timed_out() {
            FollowSignal::Signalled
        } else {
            FollowSignal::WaitElapsed
        }
    }
}

// Integration tests against a real, ephemeral Redis via testcontainers.
// Ignored by default so `cargo test --workspace` needs no Docker; run with:
//   cargo test -p zerocache-adapters-redis --lib coordinator -- --ignored
#[cfg(test)]
mod live_coordinator_tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use testcontainers_modules::{
        redis::{Redis, REDIS_PORT},
        testcontainers::{runners::SyncRunner, Container},
    };
    use zerocache_core::CacheKey;
    use zerocache_ports::{CoalescingCoordinator, FollowSignal, Role};

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
        // a's lock is short so it expires within the sleep below; b's is longer
        // so it comfortably outlives the c.try_lead a few ms later.
        let a = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(1)).unwrap();
        let b = RedisCoordinator::connect_with_lock_ttl(&url, Duration::from_secs(2)).unwrap();
        // Built up front so no pool/reader work sits between b.try_lead and
        // c.try_lead -- only a.complete does, keeping b's lock unambiguously
        // live when c contends.
        let c = RedisCoordinator::connect(&url).unwrap();
        let k = key(4);
        assert_eq!(a.try_lead(&k), Role::Leader);
        std::thread::sleep(Duration::from_millis(1300)); // a's 1s lock expires
        assert_eq!(b.try_lead(&k), Role::Leader); // b now owns it (2s TTL)
        a.complete(&k); // non-owner: the check-and-del guard must spare b's lock
        assert_eq!(c.try_lead(&k), Role::Follower); // b's lock still held
    }

    #[test]
    #[ignore]
    fn a_follower_is_woken_when_the_leader_completes() {
        let (_c, url) = start_redis();
        let leader = RedisCoordinator::connect(&url).unwrap();
        let follower = Arc::new(RedisCoordinator::connect(&url).unwrap());
        let k = key(10);
        assert_eq!(leader.try_lead(&k), Role::Leader);
        assert_eq!(follower.try_lead(&k), Role::Follower);

        let f = Arc::clone(&follower);
        let handle = thread::spawn(move || f.follow(&key(10), Duration::from_secs(5)));
        thread::sleep(Duration::from_millis(200));
        leader.complete(&k);

        assert_eq!(
            handle.join().unwrap(),
            FollowSignal::Signalled,
            "follow must return Signalled once the leader completes"
        );
    }

    #[test]
    #[ignore]
    fn follow_times_out_to_wait_elapsed_when_nothing_is_published() {
        let (_c, url) = start_redis();
        let follower = RedisCoordinator::connect(&url).unwrap();
        let k = key(11);
        let started = std::time::Instant::now();
        assert_eq!(
            follower.follow(&k, Duration::from_millis(300)),
            FollowSignal::WaitElapsed
        );
        assert!(started.elapsed() >= Duration::from_millis(250));
    }

    #[test]
    #[ignore]
    fn connect_returns_only_after_the_reader_is_subscribed() {
        let (_c, url) = start_redis();
        let publisher = RedisCoordinator::connect(&url).unwrap();
        let follower = RedisCoordinator::connect(&url).unwrap();
        let k = key(20);
        // Pre-register the wake target (follow() does this first anyway) so the
        // message has somewhere to land regardless of ordering.
        follower.slot_for(&k);
        // Publish immediately -- no 200ms pre-sleep. This is only delivered if
        // connect() returned with the reader already psubscribed.
        publisher.complete(&k);
        assert_eq!(
            follower.follow(&k, Duration::from_secs(5)),
            FollowSignal::Signalled,
            "connect() returning must imply the pub/sub reader is subscribed"
        );
    }

    #[test]
    #[ignore]
    fn two_followers_on_one_replica_both_wake_for_the_same_key() {
        let (_c, url) = start_redis();
        let leader = RedisCoordinator::connect(&url).unwrap();
        let follower = Arc::new(RedisCoordinator::connect(&url).unwrap());
        let k = key(12);
        leader.try_lead(&k);

        let f1 = Arc::clone(&follower);
        let f2 = Arc::clone(&follower);
        let h1 = thread::spawn(move || f1.follow(&key(12), Duration::from_secs(5)));
        let h2 = thread::spawn(move || f2.follow(&key(12), Duration::from_secs(5)));
        thread::sleep(Duration::from_millis(200));
        leader.complete(&k);
        assert_eq!(h1.join().unwrap(), FollowSignal::Signalled);
        assert_eq!(h2.join().unwrap(), FollowSignal::Signalled);
    }
}

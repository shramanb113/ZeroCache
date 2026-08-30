//! Cross-replica single-flight: wrap one single-key provider call so that two
//! replicas missing on the same `CacheKey` share one upstream call. Sits
//! outside the in-process `futures::Shared` coalescing (items 11/20/21) but
//! inside the shared future, so in-process piggybackers benefit too.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use zerocache_core::CacheKey;
use zerocache_ports::{CoalescingCoordinator, FollowSignal, Role};

use crate::app::AppError;

/// Whether a resolved value came from this replica's own provider call or was
/// filled by a peer replica while we waited. Only `completion.rs` cares:
/// `FromPeer` is a cache hit (record tokens saved, do not re-store); `Local`
/// is a miss handled exactly as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Coalesced {
    Local,
    FromPeer,
}

pub(crate) enum CrossReplica<T> {
    /// We ran the provider call (led, promoted, or fell back). Caller stores it.
    Led(T),
    /// A peer filled the entry; `T` was read back from the store. No tokens.
    Followed(T),
}

#[derive(Clone, Copy)]
pub(crate) struct CoalesceTiming {
    pub deadline: Duration,
    pub poll: Duration,
}

impl CoalesceTiming {
    pub(crate) const PROD: CoalesceTiming = CoalesceTiming {
        deadline: Duration::from_secs(30),
        poll: Duration::from_millis(250),
    };
}

/// In-process-only coordinator: always leads, so `coalesce_cross_replica`
/// collapses to `fetch()` plus a no-op lock pair.
pub(crate) struct NoopCoordinator;

impl CoalescingCoordinator for NoopCoordinator {
    fn try_lead(&self, _key: &CacheKey) -> Role {
        Role::Leader
    }
    fn complete(&self, _key: &CacheKey) {}
    fn follow(&self, _key: &CacheKey, _wait: Duration) -> FollowSignal {
        FollowSignal::WaitElapsed
    }
}

async fn spawn_try_lead(coord: &Arc<dyn CoalescingCoordinator>, key: CacheKey) -> Role {
    let c = Arc::clone(coord);
    tokio::task::spawn_blocking(move || c.try_lead(&key))
        .await
        .expect("coordinator try_lead task panicked")
}

async fn spawn_complete(coord: &Arc<dyn CoalescingCoordinator>, key: CacheKey) {
    let c = Arc::clone(coord);
    tokio::task::spawn_blocking(move || c.complete(&key))
        .await
        .expect("coordinator complete task panicked");
}

async fn spawn_follow(
    coord: &Arc<dyn CoalescingCoordinator>,
    key: CacheKey,
    wait: Duration,
) -> FollowSignal {
    let c = Arc::clone(coord);
    tokio::task::spawn_blocking(move || c.follow(&key, wait))
        .await
        .expect("coordinator follow task panicked")
}

/// Runs one single-key fetch under cross-replica single-flight.
///
/// `read` returns `Ok(Some(value))` once some replica has filled `key`,
/// `Ok(None)` while it is still absent (including a transient inability to
/// check -- the deadline fallback covers a persistently broken store).
/// `fetch` performs the real upstream call.
pub(crate) async fn coalesce_cross_replica<T, RFut, FFut>(
    coordinator: &Arc<dyn CoalescingCoordinator>,
    key: CacheKey,
    timing: CoalesceTiming,
    read: impl Fn() -> RFut,
    fetch: impl FnOnce() -> FFut,
) -> Result<CrossReplica<T>, AppError>
where
    RFut: Future<Output = Result<Option<T>, AppError>>,
    FFut: Future<Output = Result<T, AppError>>,
{
    match spawn_try_lead(coordinator, key).await {
        Role::Leader => {
            // Release + PUBLISH on failure too, or the lock sits for its full
            // TTL with no signal and every peer waits out its own deadline.
            let result = fetch().await;
            spawn_complete(coordinator, key).await;
            Ok(CrossReplica::Led(result?))
        }
        Role::Follower => {
            let deadline = Instant::now() + timing.deadline;
            loop {
                if let Some(value) = read().await? {
                    return Ok(CrossReplica::Followed(value));
                }
                if Instant::now() >= deadline {
                    break;
                }
                if spawn_follow(coordinator, key, timing.poll).await == FollowSignal::Signalled {
                    // The leader is done. One more read; still absent means it
                    // stored nothing (non-2xx or a transport error), so promote
                    // now instead of waiting out the deadline.
                    if let Some(value) = read().await? {
                        return Ok(CrossReplica::Followed(value));
                    }
                    break;
                }
            }
            match spawn_try_lead(coordinator, key).await {
                Role::Leader => {
                    let result = fetch().await;
                    spawn_complete(coordinator, key).await;
                    Ok(CrossReplica::Led(result?))
                }
                Role::Follower => Ok(CrossReplica::Led(fetch().await?)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::Mutex;

    use super::*;

    const FAST: CoalesceTiming = CoalesceTiming {
        deadline: Duration::from_millis(300),
        poll: Duration::from_millis(15),
    };

    /// Returns a scripted `Role` per `try_lead` call (last entry repeats),
    /// records `complete` calls, and lets a test fire `Signalled`.
    struct MockCoordinator {
        roles: Vec<Role>,
        try_lead_calls: AtomicUsize,
        complete_calls: AtomicUsize,
        follow_calls: AtomicUsize,
        signal: Mutex<Receiver<()>>,
        /// `follow` returns `Signalled` at once, every time -- what a leader
        /// that published `done` and stored nothing looks like to a follower.
        always_signal: bool,
    }

    impl MockCoordinator {
        fn new(roles: Vec<Role>) -> (Arc<Self>, Sender<()>) {
            Self::build(roles, false)
        }

        fn new_always_signalling(roles: Vec<Role>) -> (Arc<Self>, Sender<()>) {
            Self::build(roles, true)
        }

        fn build(roles: Vec<Role>, always_signal: bool) -> (Arc<Self>, Sender<()>) {
            let (tx, rx) = std::sync::mpsc::channel();
            let me = Arc::new(Self {
                roles,
                try_lead_calls: AtomicUsize::new(0),
                complete_calls: AtomicUsize::new(0),
                follow_calls: AtomicUsize::new(0),
                signal: Mutex::new(rx),
                always_signal,
            });
            (me, tx)
        }
    }

    impl CoalescingCoordinator for MockCoordinator {
        fn try_lead(&self, _key: &CacheKey) -> Role {
            let n = self.try_lead_calls.fetch_add(1, Ordering::SeqCst);
            *self
                .roles
                .get(n)
                .unwrap_or_else(|| self.roles.last().unwrap())
        }
        fn complete(&self, _key: &CacheKey) {
            self.complete_calls.fetch_add(1, Ordering::SeqCst);
        }
        fn follow(&self, _key: &CacheKey, wait: Duration) -> FollowSignal {
            self.follow_calls.fetch_add(1, Ordering::SeqCst);
            if self.always_signal {
                return FollowSignal::Signalled;
            }
            match self.signal.lock().unwrap().recv_timeout(wait) {
                Ok(()) => FollowSignal::Signalled,
                Err(_) => FollowSignal::WaitElapsed,
            }
        }
    }

    fn key() -> CacheKey {
        CacheKey::from_bytes([9u8; 32])
    }

    #[tokio::test]
    async fn leader_calls_fetch_once_then_complete() {
        let (coord, _tx) = MockCoordinator::new(vec![Role::Leader]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();
        let fetches = AtomicUsize::new(0);

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async { Ok::<Option<u32>, AppError>(None) },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(7)
            },
        )
        .await
        .unwrap();

        assert!(matches!(out, CrossReplica::Led(7)));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert_eq!(coord.complete_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_leader_whose_fetch_errors_still_releases_the_lock() {
        let (coord, _tx) = MockCoordinator::new(vec![Role::Leader]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async { Ok::<Option<u32>, AppError>(None) },
            || async {
                Err::<u32, AppError>(AppError::Provider(zerocache_ports::ProviderError(
                    "upstream unreachable".into(),
                )))
            },
        )
        .await;

        assert!(matches!(out, Err(AppError::Provider(_))));
        assert_eq!(coord.complete_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn follower_whose_read_fills_returns_followed_without_fetching() {
        let (coord, _tx) = MockCoordinator::new(vec![Role::Follower]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();
        let reads = AtomicUsize::new(0);
        let fetches = AtomicUsize::new(0);

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async {
                let n = reads.fetch_add(1, Ordering::SeqCst);
                Ok::<Option<u32>, AppError>(if n == 0 { None } else { Some(42) })
            },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(0)
            },
        )
        .await
        .unwrap();

        assert!(matches!(out, CrossReplica::Followed(42)));
        assert_eq!(fetches.load(Ordering::SeqCst), 0);
        assert_eq!(coord.complete_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn follower_signalled_with_the_entry_still_absent_promotes_without_spinning() {
        let (coord, _tx) =
            MockCoordinator::new_always_signalling(vec![Role::Follower, Role::Leader]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();
        let reads = AtomicUsize::new(0);
        let fetches = AtomicUsize::new(0);
        let started = Instant::now();

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async {
                reads.fetch_add(1, Ordering::SeqCst);
                Ok::<Option<u32>, AppError>(None)
            },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(3)
            },
        )
        .await
        .unwrap();

        // A leader that published `done` and stored nothing (non-2xx or a
        // transport error): promote at once instead of polling to the deadline.
        assert!(matches!(out, CrossReplica::Led(3)));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert_eq!(coord.complete_calls.load(Ordering::SeqCst), 1);
        assert!(
            reads.load(Ordering::SeqCst) <= 3,
            "expected one poll pass, got {} reads",
            reads.load(Ordering::SeqCst)
        );
        assert!(
            coord.follow_calls.load(Ordering::SeqCst) <= 2,
            "expected one poll pass, got {} follows",
            coord.follow_calls.load(Ordering::SeqCst)
        );
        assert!(started.elapsed() < FAST.deadline, "waited out the deadline");
    }

    #[tokio::test]
    async fn follower_that_never_sees_a_fill_falls_back_to_one_fetch() {
        let (coord, _tx) = MockCoordinator::new(vec![Role::Follower]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();
        let fetches = AtomicUsize::new(0);

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async { Ok::<Option<u32>, AppError>(None) },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(5)
            },
        )
        .await
        .unwrap();

        assert!(matches!(out, CrossReplica::Led(5)));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        // try_lead: once up front, once at the deadline
        assert_eq!(coord.try_lead_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_follower_promoted_at_the_deadline_fetches_and_completes() {
        let (coord, _tx) = MockCoordinator::new(vec![Role::Follower, Role::Leader]);
        let c: Arc<dyn CoalescingCoordinator> = coord.clone();
        let fetches = AtomicUsize::new(0);

        let out = coalesce_cross_replica(
            &c,
            key(),
            FAST,
            || async { Ok::<Option<u32>, AppError>(None) },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(1)
            },
        )
        .await
        .unwrap();

        assert!(matches!(out, CrossReplica::Led(1)));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert_eq!(coord.complete_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn noop_coordinator_is_a_pass_through_to_fetch() {
        let c: Arc<dyn CoalescingCoordinator> = Arc::new(NoopCoordinator);
        let reads = AtomicUsize::new(0);
        let fetches = AtomicUsize::new(0);

        let out = coalesce_cross_replica(
            &c,
            key(),
            CoalesceTiming::PROD,
            || async {
                reads.fetch_add(1, Ordering::SeqCst);
                Ok::<Option<u32>, AppError>(None)
            },
            || async {
                fetches.fetch_add(1, Ordering::SeqCst);
                Ok::<u32, AppError>(99)
            },
        )
        .await
        .unwrap();

        assert!(matches!(out, CrossReplica::Led(99)));
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert_eq!(
            reads.load(Ordering::SeqCst),
            0,
            "leader path never reads the store"
        );
    }
}

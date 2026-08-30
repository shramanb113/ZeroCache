use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::{FutureExt, Shared};
use prometheus::{Encoder, IntCounterVec, Opts, Registry, TextEncoder};
use tracing::Instrument;

use zerocache_core::{canonicalize_text, normalize_text, reconcile, CacheKey};
use zerocache_ports::ImageInput;
use zerocache_ports::{
    ChatCompletionProvider, ChatCompletionResponse, CompletionStore, CompletionUsage,
    EmbeddingProvider, EmbeddingStore, ImageEmbeddingProvider, ProviderError, ProviderUsage,
    StoreError,
};

/// What a coalesced fetch resolves to: every claimed key's vector plus the
/// usage from the one real provider call that produced them. `Arc`-wrapped
/// so every waiter's clone of the `Shared` future is cheap regardless of
/// batch size; `ProviderUsage` is already `Copy`, `ProviderError` is
/// `Clone` (zerocache-ports) specifically so this type can satisfy
/// `Shared`'s `Output: Clone` bound.
type SharedFetchOutput = Result<(Arc<HashMap<CacheKey, Vec<f32>>>, ProviderUsage), ProviderError>;
type SharedFetch = Shared<Pin<Box<dyn Future<Output = SharedFetchOutput> + Send>>>;

/// What a coalesced *completion* fetch resolves to: the single shared
/// upstream response plus whether a peer replica filled it (cross-replica
/// coalescing). `Arc`-wrapped so every waiter's clone is cheap;
/// `ChatCompletionResponse`, `Coalesced`, and `ProviderError` are all
/// `Clone`, satisfying `Shared`'s `Output: Clone` bound. Consumed by
/// `crate::completion::fetch_completion_coalesced`.
pub(crate) type SharedCompletionOutput =
    Result<(Arc<ChatCompletionResponse>, crate::coalesce::Coalesced), ProviderError>;
pub(crate) type SharedCompletion =
    Shared<Pin<Box<dyn Future<Output = SharedCompletionOutput> + Send>>>;

// A store call is not bounded by PROVIDER_TIMEOUT, which only applies to
// provider HTTP calls -- without this, a stuck store backend (e.g. a stale
// Redis connection) could hang a request indefinitely, and /ready, which
// exists specifically to detect exactly that, would hang right along with
// it, defeating the point of a readiness probe. Bounding every store call
// uniformly here (not just in the Redis adapter) means a hung backend of
// any kind degrades into a fast, visible AppError::Store instead of an
// indefinite hang -- one that would otherwise also block graceful
// shutdown's drain, since with_graceful_shutdown waits for in-flight
// handler futures, and an unbounded store call is awaited inside one.
const STORE_TIMEOUT: Duration = Duration::from_secs(5);

/// Runs a blocking store operation on the blocking thread pool, bounded by
/// `STORE_TIMEOUT`. Note this bounds how long the *caller* waits, not the
/// spawned OS thread itself -- a genuinely wedged sync call still occupies a
/// blocking-pool thread until it eventually returns (Rust cannot preempt
/// blocking code). The Redis adapter additionally sets its own socket-level
/// read/write timeouts, which do bound the underlying call itself; this
/// wrapper is the uniform backstop for any backend, including ones that
/// don't set their own.
///
/// When `task` loops over multiple keys (as `embed_batch`'s reconcile and
/// write-back steps, and `delete_batch`, all do), `STORE_TIMEOUT` bounds the
/// whole loop, not each individual key -- a per-batch budget, not a
/// per-operation one. Fine given `MAX_BATCH_SIZE = 100`, but worth knowing
/// if that constant ever grows.
pub(crate) async fn run_store_task<T, F>(task: F) -> Result<T, AppError>
where
    F: FnOnce() -> Result<T, AppError> + Send + 'static,
    T: Send + 'static,
{
    run_store_task_with_timeout(STORE_TIMEOUT, task).await
}

/// The actual timeout/panic-disambiguation logic, parameterized so tests can
/// exercise the timeout-fires and panic-propagates branches in milliseconds
/// instead of waiting on the real `STORE_TIMEOUT`.
async fn run_store_task_with_timeout<T, F>(timeout: Duration, task: F) -> Result<T, AppError>
where
    F: FnOnce() -> Result<T, AppError> + Send + 'static,
    T: Send + 'static,
{
    tokio::time::timeout(timeout, tokio::task::spawn_blocking(task))
        .await
        .map_err(|_| AppError::Store(StoreError("store operation timed out".into())))?
        .expect("store task panicked")
}

pub struct AppState {
    pub store: Arc<dyn EmbeddingStore>,
    pub providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    // Only ever populated with "gemini" -- see CLAUDE.md Deviations: OpenAI
    // has no public image-embedding API, so this map is deliberately a
    // strict subset of `providers`, not a parallel full registry.
    pub image_providers: HashMap<String, Arc<dyn ImageEmbeddingProvider>>,
    pub metrics: Metrics,
    // Request coalescing (in-process only -- see fetch_coalesced): tracks
    // provider fetches currently in flight, keyed by CacheKey, so two
    // concurrent requests missing on the exact same key share one provider
    // call instead of each independently paying for it. A std Mutex, not
    // tokio's: the critical section is pure HashMap bookkeeping, never
    // holds the lock across an .await, so there's no reason to pay for an
    // async-aware mutex. pub for the same reason every other AppState field
    // is: this codebase constructs AppState via plain struct literals, not
    // a constructor, everywhere it's built (main.rs, tests). Text-only --
    // `image_in_flight` below is the image-embedding counterpart.
    pub in_flight: Mutex<HashMap<CacheKey, SharedFetch>>,
    // The image-embedding counterpart to `in_flight`, used by
    // fetch_image_coalesced. A separate map, not a shared one: an image and
    // a text input can never collide on CacheKey (derive_image is
    // domain-separated from derive), but keeping the maps apart also means
    // a burst of image traffic can never contend the text path's lock, and
    // vice versa. Same SharedFetch/SharedFetchOutput type aliases as
    // `in_flight` -- both just map CacheKey -> resolved vectors + usage,
    // regardless of whether the fetch behind them embedded text or images.
    pub image_in_flight: Mutex<HashMap<CacheKey, SharedFetch>>,
    // --- semantic LLM completion cache (see crate::completion) ---
    // Opaque byte store for cached chat-completion responses. A sled/redis
    // adapter implements both this and `store` on one struct, so in
    // main.rs this is a second trait-object Arc over the very same backend
    // instance, never a second open handle.
    pub completion_store: Arc<dyn CompletionStore>,
    // Keyed by `{provider}` path segment, exactly like `providers`. Only the
    // OpenAI-shaped chat adapter is registered today; an unknown or
    // unregistered provider 404s straight out of a missing map key.
    pub completion_providers: HashMap<String, Arc<dyn ChatCompletionProvider>>,
    // The completion counterpart to `in_flight`: concurrent requests missing
    // on the exact same completion CacheKey share one upstream call. A
    // separate map for the same reason `image_in_flight` is separate --
    // domain-separated keys can't collide, and independent locks keep one
    // path's traffic from contending another's.
    pub completion_in_flight: Mutex<HashMap<CacheKey, SharedCompletion>>,
    // Cross-replica single-flight (see crate::coalesce). NoopCoordinator
    // unless ZEROCACHE_CROSS_REPLICA_COALESCING=1 on the redis backend.
    pub coordinator: Arc<dyn zerocache_ports::CoalescingCoordinator>,
    // Semantic completion tier; None when disabled or on the redis backend.
    #[cfg(feature = "semantic")]
    pub semantic: Option<crate::semantic::SemanticState>,
}

// Cumulative counters since process start, in Prometheus text-exposition
// format at GET /metrics, labeled by provider. Deliberately minimal — just
// what Phase 1's success criteria (measured hit rate, measured tokens
// billed) need, not the full PRD §11 spec. No owner/tenant label: that would
// leak tenant identity into a monitoring system and create unbounded
// cardinality (one series per tenant). Provider is small and bounded.
pub struct Metrics {
    registry: Registry,
    hits: IntCounterVec,
    misses: IntCounterVec,
    prompt_tokens_billed: IntCounterVec,
    // Completion cache (crate::completion). Labeled by `provider` only -- no
    // content_type split, since these are chat completions, not embeddings.
    // "tokens saved" (not "billed"): on a hit we know exactly what the
    // provider would have charged, from the stored response's own usage
    // block, so this is the direct money-saved number.
    completion_hits: IntCounterVec,
    completion_misses: IntCounterVec,
    // Semantic subset of completion_hits (a semantic hit bumps both).
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    completion_semantic_hits: IntCounterVec,
    completion_prompt_tokens_saved: IntCounterVec,
    completion_completion_tokens_saved: IntCounterVec,
    // A request served from a peer replica's fill without this replica's own
    // provider call (crate::coalesce). `kind` = "embedding" | "completion".
    cross_replica_coalesced: IntCounterVec,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    semantic_index_events_applied: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let hits = IntCounterVec::new(
            Opts::new(
                "zerocache_cache_hits_total",
                "Total embedding requests served from cache without a provider call",
            ),
            &["provider", "content_type"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let misses = IntCounterVec::new(
            Opts::new(
                "zerocache_cache_misses_total",
                "Total embedding requests that required a provider call",
            ),
            &["provider", "content_type"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let prompt_tokens_billed = IntCounterVec::new(
            Opts::new(
                "zerocache_provider_prompt_tokens_total",
                "Total prompt tokens billed by the embedding provider",
            ),
            &["provider", "content_type"],
        )
        .expect("metric name/help/labels are hardcoded and valid");

        let completion_hits = IntCounterVec::new(
            Opts::new(
                "zerocache_completion_cache_hits_total",
                "Total chat-completion requests served from cache without a provider call",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let completion_misses = IntCounterVec::new(
            Opts::new(
                "zerocache_completion_cache_misses_total",
                "Total chat-completion requests that required a provider call",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let completion_semantic_hits = IntCounterVec::new(
            Opts::new(
                "zerocache_completion_semantic_hits_total",
                "Chat-completion requests served via a semantic near-match (subset of zerocache_completion_cache_hits_total)",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let completion_prompt_tokens_saved = IntCounterVec::new(
            Opts::new(
                "zerocache_completion_prompt_tokens_saved_total",
                "Prompt tokens not billed by the provider because a completion was served from cache",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let completion_completion_tokens_saved = IntCounterVec::new(
            Opts::new(
                "zerocache_completion_completion_tokens_saved_total",
                "Completion tokens not billed by the provider because a completion was served from cache",
            ),
            &["provider"],
        )
        .expect("metric name/help/labels are hardcoded and valid");
        let cross_replica_coalesced = IntCounterVec::new(
            Opts::new(
                "zerocache_cross_replica_coalesced_total",
                "Requests served from a peer replica's fill without this replica's own provider call",
            ),
            &["provider", "kind"],
        )
        .expect("metric name/help/labels are hardcoded and valid");

        let semantic_index_events_applied = IntCounterVec::new(
            Opts::new(
                "zerocache_semantic_index_events_applied_total",
                "Semantic-index change-feed events applied to this replica's in-memory index",
            ),
            &["op"],
        )
        .expect("metric name/help/labels are hardcoded and valid");

        for collector in [
            &hits,
            &misses,
            &prompt_tokens_billed,
            &completion_hits,
            &completion_misses,
            &completion_semantic_hits,
            &completion_prompt_tokens_saved,
            &completion_completion_tokens_saved,
            &cross_replica_coalesced,
            &semantic_index_events_applied,
        ] {
            registry
                .register(Box::new(collector.clone()))
                .expect("registering a metric once on a fresh registry cannot fail");
        }

        Self {
            registry,
            hits,
            misses,
            prompt_tokens_billed,
            completion_hits,
            completion_misses,
            completion_semantic_hits,
            completion_prompt_tokens_saved,
            completion_completion_tokens_saved,
            cross_replica_coalesced,
            semantic_index_events_applied,
        }
    }

    fn record(&self, provider: &str, content_type: &str, stats: &BatchStats) {
        self.hits
            .with_label_values(&[provider, content_type])
            .inc_by(stats.hits as u64);
        self.misses
            .with_label_values(&[provider, content_type])
            .inc_by(stats.misses as u64);
        self.prompt_tokens_billed
            .with_label_values(&[provider, content_type])
            .inc_by(stats.provider_prompt_tokens as u64);
    }

    /// A completion served from cache: one hit, plus the prompt/completion
    /// tokens the caller was not billed for (taken from the stored
    /// response's own usage block).
    pub(crate) fn record_completion_hit(&self, provider: &str, usage: &CompletionUsage) {
        self.completion_hits.with_label_values(&[provider]).inc();
        self.completion_prompt_tokens_saved
            .with_label_values(&[provider])
            .inc_by(usage.prompt_tokens as u64);
        self.completion_completion_tokens_saved
            .with_label_values(&[provider])
            .inc_by(usage.completion_tokens as u64);
    }

    /// A completion that required a provider call (including one that
    /// piggybacked on a coalesced in-flight fetch -- it still wasn't in the
    /// store).
    pub(crate) fn record_completion_miss(&self, provider: &str) {
        self.completion_misses.with_label_values(&[provider]).inc();
    }

    /// A completion rescued by the semantic tier. The caller also calls
    /// `record_completion_hit`; this counter is that total's semantic slice.
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub(crate) fn record_completion_semantic_hit(&self, provider: &str) {
        self.completion_semantic_hits
            .with_label_values(&[provider])
            .inc();
    }

    /// A request that avoided its own provider call because a peer replica
    /// filled the entry first (crate::coalesce). `kind` = "embedding" |
    /// "completion".
    pub(crate) fn record_cross_replica_coalesced(&self, provider: &str, kind: &str) {
        self.cross_replica_coalesced
            .with_label_values(&[provider, kind])
            .inc();
    }

    /// A cheap clone of the counter vec, for incrementing from inside a
    /// 'static coalesced future that cannot borrow `&Metrics`. Consumed by
    /// `fetch_coalesced`'s single-key cross-replica path.
    pub(crate) fn cross_replica_coalesced_counter(&self) -> IntCounterVec {
        self.cross_replica_coalesced.clone()
    }

    /// Change-feed events (`upsert` / `delete`) the poll task applied to the
    /// local in-memory semantic index this tick.
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub(crate) fn record_semantic_index_events_applied(&self, upserts: u64, deletes: u64) {
        if upserts > 0 {
            self.semantic_index_events_applied
                .with_label_values(&["upsert"])
                .inc_by(upserts);
        }
        if deletes > 0 {
            self.semantic_index_events_applied
                .with_label_values(&["delete"])
                .inc_by(deletes);
        }
    }

    /// Renders all registered metrics in Prometheus text exposition format.
    pub fn encode(&self) -> String {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&metric_families, &mut buffer)
            .expect("encoding already-gathered metrics cannot fail");
        String::from_utf8(buffer).expect("prometheus text encoder always emits valid utf8")
    }
}

#[derive(Debug)]
pub enum AppError {
    Store(StoreError),
    Provider(ProviderError),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Store(e) => write!(f, "{e}"),
            AppError::Provider(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AppError {}

pub struct BatchStats {
    pub hits: usize,
    pub misses: usize,
    pub provider_prompt_tokens: u32,
    pub provider_total_tokens: u32,
}

/// Everything one `/{provider}/v1/embeddings` call needs to run the cache
/// flow: which adapter to call, what to call it (for the cache key and the
/// metrics label — kept separate from the adapter reference itself since the
/// registry is keyed by name, not by adapter identity), whose namespace this
/// is, and the batch itself.
pub struct EmbedRequest<'a> {
    // Owned (Arc), not borrowed like the other fields: fetch_coalesced
    // needs to build a 'static future that can outlive this request's own
    // stack frame when another concurrent request piggybacks on it.
    pub provider: Arc<dyn EmbeddingProvider>,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub texts: &'a [String],
}

/// Runs the cache flow for one batch: reconcile against the store (scoped to
/// `request.owner_id` + `request.provider_name` via the cache key), fetch
/// only the misses from `request.provider`, write them back, and return
/// vectors in the same order as `request.texts`.
///
/// Text is normalized (trimmed, internal whitespace collapsed) before both
/// hashing and being sent to the provider — applying it in only one place
/// would let the cache key and what's actually embedded silently diverge.
/// Misses that normalize to an identical string are deduplicated: the
/// provider is called once per unique text, not once per original index, and
/// the same fetched vector is broadcast back to every matching position.
///
/// Store reads and store writes each run via `run_store_task`, which puts
/// them on their own `tokio::task::spawn_blocking` (since `sled`/`redis` are
/// synchronous) bounded by `STORE_TIMEOUT`. The provider call in between is
/// awaited directly — it's async now, not blocking, so it runs on the same
/// runtime as the rest of the server instead of needing a thread-pool hop,
/// and is bounded separately by each adapter's own `PROVIDER_TIMEOUT`.
///
/// A store or provider failure aborts the whole batch rather than degrading
/// silently into an extra miss or a dropped write — the caller is expected
/// to surface it as an error response.
#[tracing::instrument(skip_all, fields(provider = %request.provider_name, texts = request.texts.len(), hits, misses))]
pub async fn embed_batch(
    state: &AppState,
    request: EmbedRequest<'_>,
) -> Result<(Vec<Vec<f32>>, BatchStats), AppError> {
    let provider_version = request.provider.version();
    // Resolved once per batch, not once per text: for a cloud adapter this
    // parses the caller's model string, which is pure but not free, and the
    // answer is identical for every text in the batch.
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    // Two derived forms of each input, deliberately split:
    //  - `provider_texts` (whitespace-only `normalize_text`) is what actually
    //    gets embedded, so a stored vector is always a real embedding of a
    //    real caller's actual text.
    //  - `key_texts` (`canonicalize_text`: also case-folded, NFC'd, quote/dash
    //    -folded, trailing-punctuation-trimmed) is what the cache key hashes,
    //    so inputs that differ only in those respects share one entry. The
    //    first miss in such a group wins: its `provider_texts` form is the one
    //    embedded and stored for the whole group.
    let provider_texts: Vec<String> = request
        .texts
        .iter()
        .map(|text| normalize_text(text))
        .collect();
    let key_texts: Vec<String> = request
        .texts
        .iter()
        .map(|text| canonicalize_text(text))
        .collect();
    let keys: Vec<CacheKey> = key_texts
        .iter()
        .map(|text| {
            CacheKey::derive(
                request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
                provider_version,
                text,
            )
        })
        .collect();

    let reconciled = {
        let store = Arc::clone(&state.store);
        let keys_for_lookup = keys.clone();
        run_store_task(move || {
            reconcile(&keys_for_lookup, |key| {
                store.get(key).map_err(AppError::Store)
            })
        })
        .instrument(tracing::info_span!("store_lookup"))
        .await?
    };

    let hits = reconciled.hits.len();
    let misses = reconciled.misses.len();
    tracing::Span::current()
        .record("hits", hits)
        .record("misses", misses);

    let mut results: Vec<Option<Vec<f32>>> = vec![None; request.texts.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    let mut provider_prompt_tokens = 0;
    let mut provider_total_tokens = 0;

    if !reconciled.misses.is_empty() {
        // Dedupe misses by CacheKey: several original indices can normalize
        // to the identical text (or already be literal duplicates in the
        // input), and each such group shares one CacheKey. Fetch each
        // unique text once, then broadcast its vector to every index that
        // shares its key.
        let mut unique_miss_texts: Vec<String> = Vec::new();
        let mut unique_keys: Vec<CacheKey> = Vec::new();
        let mut key_to_unique_index: HashMap<CacheKey, usize> = HashMap::new();
        let mut index_to_unique_index: Vec<(usize, usize)> =
            Vec::with_capacity(reconciled.misses.len());

        for (index, key) in &reconciled.misses {
            let unique_index = *key_to_unique_index.entry(*key).or_insert_with(|| {
                unique_miss_texts.push(provider_texts[*index].clone());
                unique_keys.push(*key);
                unique_miss_texts.len() - 1
            });
            index_to_unique_index.push((*index, unique_index));
        }

        // Cross-request coalescing: if another concurrent request is
        // already fetching one of our unique missing keys, piggyback on
        // its result instead of independently calling the provider for the
        // same text. See fetch_coalesced for the mechanics.
        let (fetched, usage) = fetch_coalesced(
            state,
            Arc::clone(&request.provider),
            request.provider_name,
            request.api_key,
            request.model,
            &unique_keys,
            &unique_miss_texts,
        )
        .await?;
        provider_prompt_tokens = usage.prompt_tokens;
        provider_total_tokens = usage.total_tokens;

        for (index, unique_index) in &index_to_unique_index {
            let key = &unique_keys[*unique_index];
            results[*index] = Some(fetched[key].clone());
        }

        let mut writes = Vec::with_capacity(unique_keys.len());
        for key in &unique_keys {
            writes.push((*key, fetched[key].clone()));
        }

        let store = Arc::clone(&state.store);
        run_store_task(move || -> Result<(), AppError> {
            for (key, vector) in writes {
                store.put(key, vector).map_err(AppError::Store)?;
            }
            Ok(())
        })
        .instrument(tracing::info_span!("store_write_back"))
        .await?;
    }

    let vectors = results
        .into_iter()
        .map(|v| v.expect("every index must be filled by a hit or a miss"))
        .collect();

    let stats = BatchStats {
        hits,
        misses,
        provider_prompt_tokens,
        provider_total_tokens,
    };
    state.metrics.record(request.provider_name, "text", &stats);

    Ok((vectors, stats))
}

/// Everything one `POST /{provider}/v1/images/embeddings` call needs. Only
/// ever constructed with a `gemini` provider today (see AppState's
/// `image_providers` comment) -- kept generic over `ImageEmbeddingProvider`
/// rather than hardcoded to `GeminiProvider` so a second provider can be
/// added later without touching this struct or `embed_image_batch`.
pub struct EmbedImageRequest<'a> {
    pub provider: Arc<dyn ImageEmbeddingProvider>,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub images: &'a [ImageInput],
}

/// The image-embedding counterpart to `embed_batch`. Deliberately simpler
/// in one way, documented in CLAUDE.md Deviations: no text normalization
/// (images aren't normalized). It does dedupe misses that share a CacheKey
/// within one batch -- two images with identical bytes+mime type are rare,
/// but not rare enough to justify paying the provider twice for one, and
/// the bookkeeping is the same shape `embed_batch` already pays for text.
/// It also now coalesces across concurrent requests via
/// `fetch_image_coalesced`/`AppState.image_in_flight`, the image
/// counterpart to `embed_batch`'s `fetch_coalesced`/`AppState.in_flight`
/// (CLAUDE.md Deviations item 20) -- originally deferred when multimodal
/// embeddings first shipped, since image traffic didn't yet have a measured
/// concurrent-miss gap the way text did.
#[tracing::instrument(skip_all, fields(provider = %request.provider_name, images = request.images.len(), hits, misses))]
pub async fn embed_image_batch(
    state: &AppState,
    request: EmbedImageRequest<'_>,
) -> Result<(Vec<Vec<f32>>, BatchStats), AppError> {
    let provider_version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let keys: Vec<CacheKey> = request
        .images
        .iter()
        .map(|image| {
            CacheKey::derive_image(
                request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
                provider_version,
                &image.mime_type,
                &image.data,
            )
        })
        .collect();

    let reconciled = {
        let store = Arc::clone(&state.store);
        let keys_for_lookup = keys.clone();
        run_store_task(move || {
            reconcile(&keys_for_lookup, |key| {
                store.get(key).map_err(AppError::Store)
            })
        })
        .instrument(tracing::info_span!("store_lookup"))
        .await?
    };

    let hits = reconciled.hits.len();
    let misses = reconciled.misses.len();
    tracing::Span::current()
        .record("hits", hits)
        .record("misses", misses);

    let mut results: Vec<Option<Vec<f32>>> = vec![None; request.images.len()];
    for (index, vector) in reconciled.hits {
        results[index] = Some(vector);
    }

    let mut provider_prompt_tokens = 0;
    let mut provider_total_tokens = 0;

    if !reconciled.misses.is_empty() {
        // Dedupe misses by CacheKey: several original indices can share an
        // identical image (same bytes + mime type), and each such group
        // shares one CacheKey via CacheKey::derive_image. Fetch each unique
        // image once, then broadcast its vector to every index that shares
        // its key -- same pattern as embed_batch's text-dedup, minus
        // cross-request coalescing (see the doc comment above).
        let mut unique_miss_images: Vec<ImageInput> = Vec::new();
        let mut unique_keys: Vec<CacheKey> = Vec::new();
        let mut key_to_unique_index: HashMap<CacheKey, usize> = HashMap::new();
        let mut index_to_unique_index: Vec<(usize, usize)> =
            Vec::with_capacity(reconciled.misses.len());

        for (index, key) in &reconciled.misses {
            let unique_index = *key_to_unique_index.entry(*key).or_insert_with(|| {
                unique_miss_images.push(ImageInput {
                    mime_type: request.images[*index].mime_type.clone(),
                    data: request.images[*index].data.clone(),
                });
                unique_keys.push(*key);
                unique_miss_images.len() - 1
            });
            index_to_unique_index.push((*index, unique_index));
        }

        // Cross-request coalescing: if another concurrent request is
        // already fetching one of our unique missing keys, piggyback on
        // its result instead of independently calling the provider for the
        // same image. See fetch_image_coalesced for the mechanics.
        let (fetched, usage) = fetch_image_coalesced(
            state,
            Arc::clone(&request.provider),
            request.api_key,
            request.model,
            &unique_keys,
            &unique_miss_images,
        )
        .await?;
        provider_prompt_tokens = usage.prompt_tokens;
        provider_total_tokens = usage.total_tokens;

        for (index, unique_index) in &index_to_unique_index {
            let key = &unique_keys[*unique_index];
            results[*index] = Some(fetched[key].clone());
        }

        let mut writes = Vec::with_capacity(unique_keys.len());
        for key in &unique_keys {
            writes.push((*key, fetched[key].clone()));
        }

        let store = Arc::clone(&state.store);
        run_store_task(move || -> Result<(), AppError> {
            for (key, vector) in writes {
                store.put(key, vector).map_err(AppError::Store)?;
            }
            Ok(())
        })
        .instrument(tracing::info_span!("store_write_back"))
        .await?;
    }

    let vectors = results
        .into_iter()
        .map(|v| v.expect("every index must be filled by a hit or a miss"))
        .collect();

    let stats = BatchStats {
        hits,
        misses,
        provider_prompt_tokens,
        provider_total_tokens,
    };
    state.metrics.record(request.provider_name, "image", &stats);

    Ok((vectors, stats))
}

/// The image-embedding counterpart to `DeleteRequest`/`delete_batch`.
pub struct DeleteImageRequest<'a> {
    pub provider: &'a dyn ImageEmbeddingProvider,
    pub provider_name: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub images: &'a [ImageInput],
}

#[tracing::instrument(skip_all, fields(provider = %request.provider_name, images = request.images.len()))]
pub async fn delete_image_batch(
    state: &AppState,
    request: DeleteImageRequest<'_>,
) -> Result<usize, AppError> {
    let provider_version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let keys: Vec<CacheKey> = request
        .images
        .iter()
        .map(|image| {
            CacheKey::derive_image(
                request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
                provider_version,
                &image.mime_type,
                &image.data,
            )
        })
        .collect();

    let count = keys.len();
    let store = Arc::clone(&state.store);
    run_store_task(move || -> Result<(), AppError> {
        for key in keys {
            store.delete(&key).map_err(AppError::Store)?;
        }
        Ok(())
    })
    .instrument(tracing::info_span!("store_delete"))
    .await?;

    Ok(count)
}

/// Resolves every key in `unique_keys` to its embedding, coalescing with
/// any identical concurrent in-flight fetch from another request instead of
/// independently calling the provider for the same text twice.
///
/// This only dedupes the provider call itself -- every caller (whether it
/// claimed the fetch or piggybacked on someone else's) still writes its own
/// copy of the resolved vectors back to the store afterward. That's a
/// deliberate simplification, not an oversight: redundant writes of an
/// identical value are already an accepted correctness model in this
/// codebase (see zerocache-adapters-redis's own last-write-wins rationale
/// for concurrent replicas), so coalescing the write too isn't worth the
/// extra complexity it would add here.
///
/// The returned `ProviderUsage` reflects only the subset of `unique_keys`
/// this specific call actually caused a *new* provider call for -- keys
/// resolved by piggybacking on another request's in-flight fetch
/// contribute zero tokens, exactly like a cache hit, since no additional
/// billing happened on this call's behalf. If this call claimed nothing
/// (every key was already in flight elsewhere), it returns default (zero)
/// usage, the same as an all-hit batch.
///
/// In-process for a multi-key claim: two replicas behind a load balancer
/// each run their own `in_flight` map, so the same miss is still fetched
/// once per replica. A claim that reduces to a single key additionally goes
/// through `crate::coalesce` -- a Redis-lock single-flight across replicas --
/// when a real coordinator is configured (item 26); the coordinator works one
/// CacheKey at a time, so a multi-key claim keeps exactly the old behaviour.
#[tracing::instrument(skip_all, fields(unique_keys = unique_keys.len(), claimed, piggybacked))]
async fn fetch_coalesced(
    state: &AppState,
    provider: Arc<dyn EmbeddingProvider>,
    provider_name: &str,
    api_key: &str,
    model: &str,
    unique_keys: &[CacheKey],
    unique_texts: &[String],
) -> Result<(HashMap<CacheKey, Vec<f32>>, ProviderUsage), AppError> {
    let mut piggyback_waiters: Vec<SharedFetch> = Vec::new();
    let mut claim_keys: Vec<CacheKey> = Vec::new();
    let mut claim_texts: Vec<String> = Vec::new();
    let mut own_claim: Option<SharedFetch> = None;

    {
        let mut in_flight = state.in_flight.lock().expect("in_flight mutex poisoned");

        for (key, text) in unique_keys.iter().zip(unique_texts) {
            if let Some(existing) = in_flight.get(key) {
                piggyback_waiters.push(existing.clone());
            } else {
                claim_keys.push(*key);
                claim_texts.push(text.clone());
            }
        }

        if !claim_keys.is_empty() {
            let api_key = api_key.to_string();
            let model = model.to_string();
            let claim_keys_for_future = claim_keys.clone();
            let claim_texts_for_future = claim_texts.clone();
            let claim_count = claim_keys.len();

            // Cross-replica single-flight applies only to a lone claimed key --
            // the coordinator works one CacheKey at a time. A multi-key claim
            // keeps exactly today's behaviour.
            let single_key = (claim_keys.len() == 1).then(|| claim_keys[0]);
            let coordinator = Arc::clone(&state.coordinator);
            let store = Arc::clone(&state.store);
            let coalesced_counter = state.metrics.cross_replica_coalesced_counter();
            let provider_label = provider_name.to_string();

            let fut: Pin<Box<dyn Future<Output = SharedFetchOutput> + Send>> = Box::pin(
                async move {
                    match single_key {
                        None => {
                            let (vectors, usage) = provider
                                .embed_batch(&api_key, &model, &claim_texts_for_future)
                                .await?;
                            let by_key = claim_keys_for_future.into_iter().zip(vectors).collect();
                            Ok((Arc::new(by_key), usage))
                        }
                        Some(key) => {
                            use crate::coalesce::{
                                coalesce_cross_replica, CoalesceTiming, CrossReplica,
                            };
                            let single_text = claim_texts_for_future[0].clone();
                            let outcome = coalesce_cross_replica::<
                                (HashMap<CacheKey, Vec<f32>>, ProviderUsage),
                                _,
                                _,
                            >(
                                &coordinator,
                                key,
                                CoalesceTiming::PROD,
                                || {
                                    let store = Arc::clone(&store);
                                    async move {
                                        let got = run_store_task(move || {
                                            store.get(&key).map_err(AppError::Store)
                                        })
                                        .await
                                        .unwrap_or(None);
                                        Ok(got.map(|v| {
                                            let mut m = HashMap::with_capacity(1);
                                            m.insert(key, v);
                                            (m, ProviderUsage::default())
                                        }))
                                    }
                                },
                                {
                                    let provider = Arc::clone(&provider);
                                    let api_key = api_key.clone();
                                    let model = model.clone();
                                    let store = Arc::clone(&store);
                                    move || async move {
                                        let (vectors, usage) = provider
                                            .embed_batch(&api_key, &model, &[single_text])
                                            .await
                                            .map_err(AppError::Provider)?;
                                        let vector = vectors.into_iter().next().expect(
                                            "provider returned no vector for a one-text batch",
                                        );
                                        // Store before returning, so the
                                        // coordinator's `done` signal implies a
                                        // readable entry for peers. The caller's
                                        // write-back repeats it -- an identical
                                        // content-addressed value, and it is the
                                        // one that surfaces a store error.
                                        {
                                            let vector = vector.clone();
                                            let _ = run_store_task(move || {
                                                store.put(key, vector).map_err(AppError::Store)
                                            })
                                            .await;
                                        }
                                        let mut m = HashMap::with_capacity(1);
                                        m.insert(key, vector);
                                        Ok((m, usage))
                                    }
                                },
                            )
                            .await;

                            match outcome {
                                Ok(CrossReplica::Led((by_key, usage))) => {
                                    Ok((Arc::new(by_key), usage))
                                }
                                Ok(CrossReplica::Followed((by_key, _))) => {
                                    coalesced_counter
                                        .with_label_values(&[&provider_label, "embedding"])
                                        .inc();
                                    Ok((Arc::new(by_key), ProviderUsage::default()))
                                }
                                Err(AppError::Provider(e)) => Err(e),
                                // Defensive: the read closure maps store errors
                                // to Ok(None), so this is unreachable today.
                                Err(AppError::Store(e)) => Err(ProviderError(format!(
                                    "store error during coalesced embedding fetch: {e}"
                                ))),
                            }
                        }
                    }
                }
                .instrument(tracing::info_span!("provider_call", texts = claim_count)),
            );
            let shared: SharedFetch = fut.shared();

            for key in &claim_keys {
                in_flight.insert(*key, shared.clone());
            }
            own_claim = Some(shared);
        }
    }

    tracing::Span::current()
        .record("claimed", claim_keys.len())
        .record("piggybacked", piggyback_waiters.len());

    let mut resolved: HashMap<CacheKey, Vec<f32>> = HashMap::with_capacity(unique_keys.len());
    let mut claimed_usage = ProviderUsage::default();

    if let Some(claim_fut) = own_claim {
        let (vectors, usage) = claim_fut.await.map_err(AppError::Provider)?;
        resolved.extend(vectors.iter().map(|(k, v)| (*k, v.clone())));
        claimed_usage = usage;

        // Remove our claimed keys now that they're resolved, so a later,
        // genuinely new miss for the same key starts a fresh fetch instead
        // of piggybacking on a completed, no-longer-useful future forever.
        let mut in_flight = state.in_flight.lock().expect("in_flight mutex poisoned");
        for key in &claim_keys {
            in_flight.remove(key);
        }
    }

    for waiter in piggyback_waiters {
        let (vectors, _usage) = waiter.await.map_err(AppError::Provider)?;
        resolved.extend(vectors.iter().map(|(k, v)| (*k, v.clone())));
    }

    Ok((resolved, claimed_usage))
}

/// The image-embedding counterpart to `fetch_coalesced`, coalescing against
/// `AppState.image_in_flight` instead of `AppState.in_flight`. Identical
/// shape and semantics -- same claim-or-piggyback logic, same
/// only-the-claimer-reports-usage rule -- just
/// keyed on `ImageInput`s instead of `String` texts and calling
/// `ImageEmbeddingProvider::embed_image_batch` instead of
/// `EmbeddingProvider::embed_batch`. See `fetch_coalesced`'s doc comment for
/// the full rationale; not repeated here to avoid the two drifting out of
/// sync with each other in prose while the code itself stays in sync by
/// construction. Images are in-process only: no coordinator on this path,
/// single key or not.
#[tracing::instrument(skip_all, fields(unique_keys = unique_keys.len(), claimed, piggybacked))]
async fn fetch_image_coalesced(
    state: &AppState,
    provider: Arc<dyn ImageEmbeddingProvider>,
    api_key: &str,
    model: &str,
    unique_keys: &[CacheKey],
    unique_images: &[ImageInput],
) -> Result<(HashMap<CacheKey, Vec<f32>>, ProviderUsage), AppError> {
    let mut piggyback_waiters: Vec<SharedFetch> = Vec::new();
    let mut claim_keys: Vec<CacheKey> = Vec::new();
    let mut claim_images: Vec<ImageInput> = Vec::new();
    let mut own_claim: Option<SharedFetch> = None;

    {
        let mut image_in_flight = state
            .image_in_flight
            .lock()
            .expect("image_in_flight mutex poisoned");

        for (key, image) in unique_keys.iter().zip(unique_images) {
            if let Some(existing) = image_in_flight.get(key) {
                piggyback_waiters.push(existing.clone());
            } else {
                claim_keys.push(*key);
                claim_images.push(image.clone());
            }
        }

        if !claim_keys.is_empty() {
            let api_key = api_key.to_string();
            let model = model.to_string();
            let claim_keys_for_future = claim_keys.clone();
            let claim_images_for_future = claim_images.clone();

            let claim_count = claim_keys.len();
            let fut: Pin<Box<dyn Future<Output = SharedFetchOutput> + Send>> = Box::pin(
                async move {
                    let (vectors, usage) = provider
                        .embed_image_batch(&api_key, &model, &claim_images_for_future)
                        .await?;
                    let by_key = claim_keys_for_future.into_iter().zip(vectors).collect();
                    Ok((Arc::new(by_key), usage))
                }
                .instrument(tracing::info_span!("provider_call", images = claim_count)),
            );
            let shared: SharedFetch = fut.shared();

            for key in &claim_keys {
                image_in_flight.insert(*key, shared.clone());
            }
            own_claim = Some(shared);
        }
    }

    tracing::Span::current()
        .record("claimed", claim_keys.len())
        .record("piggybacked", piggyback_waiters.len());

    let mut resolved: HashMap<CacheKey, Vec<f32>> = HashMap::with_capacity(unique_keys.len());
    let mut claimed_usage = ProviderUsage::default();

    if let Some(claim_fut) = own_claim {
        let (vectors, usage) = claim_fut.await.map_err(AppError::Provider)?;
        resolved.extend(vectors.iter().map(|(k, v)| (*k, v.clone())));
        claimed_usage = usage;

        // Remove our claimed keys now that they're resolved, so a later,
        // genuinely new miss for the same key starts a fresh fetch instead
        // of piggybacking on a completed, no-longer-useful future forever.
        let mut image_in_flight = state
            .image_in_flight
            .lock()
            .expect("image_in_flight mutex poisoned");
        for key in &claim_keys {
            image_in_flight.remove(key);
        }
    }

    for waiter in piggyback_waiters {
        let (vectors, _usage) = waiter.await.map_err(AppError::Provider)?;
        resolved.extend(vectors.iter().map(|(k, v)| (*k, v.clone())));
    }

    Ok((resolved, claimed_usage))
}

/// Everything one `DELETE /{provider}/v1/embeddings` call needs: which
/// entries to remove, scoped to the caller's own namespace via `owner_id`
/// exactly like a normal read/write request — a caller can only ever delete
/// entries in their own namespace, never anyone else's. `provider` is only
/// used for its `version()`, to compute the same keys a matching `POST`
/// would have used; no provider call is ever made for a delete.
pub struct DeleteRequest<'a> {
    pub provider: &'a dyn EmbeddingProvider,
    pub provider_name: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub texts: &'a [String],
}

/// Deletes the cache entries that a matching `POST` with the same
/// provider/model/texts (and the same caller) would have hit. Returns the
/// number of keys the delete was attempted for -- deletion is idempotent, so
/// this is not "how many existed," just "how many were requested."
#[tracing::instrument(skip_all, fields(provider = %request.provider_name, texts = request.texts.len()))]
pub async fn delete_batch(state: &AppState, request: DeleteRequest<'_>) -> Result<usize, AppError> {
    let provider_version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let keys: Vec<CacheKey> = request
        .texts
        .iter()
        .map(|text| {
            CacheKey::derive(
                request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
                provider_version,
                // Must mirror `embed_batch`'s key derivation exactly, so a
                // DELETE removes the entry a canonical-equal POST wrote.
                &canonicalize_text(text),
            )
        })
        .collect();

    let count = keys.len();
    let store = Arc::clone(&state.store);
    run_store_task(move || -> Result<(), AppError> {
        for key in keys {
            store.delete(&key).map_err(AppError::Store)?;
        }
        Ok(())
    })
    .instrument(tracing::info_span!("store_delete"))
    .await?;

    Ok(count)
}

/// A fixed, reserved cache key used only to prove the store is reachable --
/// never a real cache entry a caller could hit or collide with, since no
/// real request can produce owner_id = [0u8; 32] (a real owner_id is always
/// derived from a hashed API key, which blake3 never maps to all-zero
/// output in practice) combined with this reserved provider/model string.
fn readiness_check_key() -> CacheKey {
    CacheKey::derive(
        [0u8; 32],
        "__zerocache_internal__",
        "__readiness_check__",
        "__readiness_check__",
        "v1",
        "",
    )
}

/// Proves the configured store backend is actually reachable, not just
/// that the process is running. A miss is a healthy, expected result (the
/// sentinel key is never written to) -- only a store-level error means
/// "not ready."
pub async fn check_store_readiness(state: &AppState) -> Result<(), AppError> {
    let store = Arc::clone(&state.store);
    let key = readiness_check_key();
    run_store_task(move || store.get(&key).map(|_| ()).map_err(AppError::Store)).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdHashMap;
    use std::sync::Mutex;

    use zerocache_ports::ProviderUsage;

    use super::*;

    #[test]
    fn record_cross_replica_coalesced_labels_by_provider_and_kind() {
        let m = Metrics::new();
        m.record_cross_replica_coalesced("openai", "completion");
        m.record_cross_replica_coalesced("openai", "completion");
        m.record_cross_replica_coalesced("gemini", "embedding");
        let dump = m.encode();
        assert!(
            dump.contains(
                "zerocache_cross_replica_coalesced_total{kind=\"completion\",provider=\"openai\"} 2"
            ),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_cross_replica_coalesced_total{kind=\"embedding\",provider=\"gemini\"} 1"
            ),
            "{dump}"
        );
    }

    #[test]
    fn record_completion_semantic_hit_increments_only_the_semantic_counter() {
        let m = Metrics::new();
        m.record_completion_semantic_hit("openai");
        m.record_completion_semantic_hit("openai");
        let dump = m.encode();
        assert!(
            dump.contains("zerocache_completion_semantic_hits_total{provider=\"openai\"} 2"),
            "{dump}"
        );
        assert!(
            !dump.contains("zerocache_completion_cache_hits_total{provider=\"openai\"} 2"),
            "{dump}"
        );
    }

    #[test]
    fn record_semantic_index_events_applied_labels_by_op_and_skips_zeroes() {
        let m = Metrics::new();
        m.record_semantic_index_events_applied(3, 1);
        m.record_semantic_index_events_applied(2, 0);
        let dump = m.encode();
        assert!(
            dump.contains("zerocache_semantic_index_events_applied_total{op=\"upsert\"} 5"),
            "{dump}"
        );
        assert!(
            dump.contains("zerocache_semantic_index_events_applied_total{op=\"delete\"} 1"),
            "{dump}"
        );
    }

    struct MockStore {
        data: Mutex<StdHashMap<CacheKey, Vec<f32>>>,
    }

    impl MockStore {
        fn empty() -> Self {
            Self {
                data: Mutex::new(StdHashMap::new()),
            }
        }

        fn with(entries: Vec<(CacheKey, Vec<f32>)>) -> Self {
            Self {
                data: Mutex::new(entries.into_iter().collect()),
            }
        }
    }

    impl EmbeddingStore for MockStore {
        fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }

        fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError> {
            self.data.lock().unwrap().insert(key, vector);
            Ok(())
        }

        fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }
    }

    struct FailingStore;

    impl EmbeddingStore for FailingStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Err(StoreError("mock store failure".into()))
        }

        fn put(&self, _key: CacheKey, _vector: Vec<f32>) -> Result<(), StoreError> {
            Err(StoreError("mock store failure".into()))
        }

        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Err(StoreError("mock store failure".into()))
        }
    }

    struct MockProvider {
        response: Vec<Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            assert_eq!(
                texts.len(),
                self.response.len(),
                "mock provider called with unexpected batch size"
            );
            Ok((
                self.response.clone(),
                ProviderUsage {
                    prompt_tokens: 10,
                    total_tokens: 10,
                },
            ))
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }

        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok("mock-scope".to_string())
        }
    }

    struct FailingProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for FailingProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            Err(ProviderError("mock provider failure".into()))
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }

        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok("mock-scope".to_string())
        }
    }

    struct PanicProvider;

    #[async_trait::async_trait]
    impl EmbeddingProvider for PanicProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            _texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            panic!("provider must not be called when the store lookup already failed");
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }

        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok("mock-scope".to_string())
        }
    }

    struct MockImageProvider {
        response: Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockImageProvider {
        fn returning(response: Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError>) -> Self {
            Self {
                response,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl ImageEmbeddingProvider for MockImageProvider {
        async fn embed_image_batch(
            &self,
            _api_key: &str,
            _model: &str,
            images: &[ImageInput],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match &self.response {
                Ok((vectors, usage)) => Ok((vectors[..images.len()].to_vec(), *usage)),
                Err(e) => Err(e.clone()),
            }
        }

        fn version(&self) -> &'static str {
            "test-image-v1"
        }

        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok("mock-scope".to_string())
        }
    }

    const OWNER_A: [u8; 32] = [1u8; 32];
    const OWNER_B: [u8; 32] = [2u8; 32];

    // The embedding-path tests don't exercise the completion cache; this
    // stand-in keeps `AppState` constructible without pulling a real
    // CompletionStore into every test.
    struct NoopCompletionStore;

    impl CompletionStore for NoopCompletionStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<u8>>, StoreError> {
            Ok(None)
        }
        fn put(&self, _key: CacheKey, _value: Vec<u8>) -> Result<(), StoreError> {
            Ok(())
        }
        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Ok(())
        }
    }

    fn state_with(store: impl EmbeddingStore + 'static) -> AppState {
        AppState {
            store: Arc::new(store),
            providers: StdHashMap::new(),
            image_providers: StdHashMap::new(),
            metrics: Metrics::new(),
            in_flight: Mutex::new(StdHashMap::new()),
            image_in_flight: Mutex::new(StdHashMap::new()),
            completion_store: Arc::new(NoopCompletionStore),
            completion_providers: StdHashMap::new(),
            completion_in_flight: Mutex::new(StdHashMap::new()),
            coordinator: Arc::new(crate::coalesce::NoopCoordinator),
            #[cfg(feature = "semantic")]
            semantic: None,
        }
    }

    #[tokio::test]
    async fn all_hits_skips_provider_entirely() {
        let key = CacheKey::derive(
            OWNER_A,
            "openai",
            "mock-scope",
            "m",
            "mock-v1",
            "cached text",
        );
        let state = state_with(MockStore::with(vec![(key, vec![1.0, 2.0])]));
        let provider = PanicProvider;
        let texts = vec!["cached text".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.provider_prompt_tokens, 0);
    }

    #[tokio::test]
    async fn all_misses_calls_provider_and_writes_back_to_store() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider {
            response: vec![vec![9.0, 9.0]],
        };
        let texts = vec!["fresh text".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![9.0, 9.0]]);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 10);

        let key = CacheKey::derive(
            OWNER_A,
            "openai",
            "mock-scope",
            "m",
            "mock-v1",
            "fresh text",
        );
        assert_eq!(state.store.get(&key).unwrap(), Some(vec![9.0, 9.0]));
    }

    #[tokio::test]
    async fn mixed_batch_preserves_original_order() {
        let hit_key = CacheKey::derive(OWNER_A, "openai", "mock-scope", "m", "mock-v1", "old");
        let state = state_with(MockStore::with(vec![(hit_key, vec![1.0])]));
        let provider = MockProvider {
            response: vec![vec![2.0]],
        };
        let texts = vec!["new".to_string(), "old".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![2.0], vec![1.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn different_owners_never_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider: Arc<dyn EmbeddingProvider> = Arc::new(MockProvider {
            response: vec![vec![7.0]],
        });
        let texts = vec!["identical text".to_string()];

        let (_, stats_a) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider),
                provider_name: "openai",
                api_key: "key-a",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_a.misses, 1, "owner A's first request must be a miss");

        let (_, stats_b) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider),
                provider_name: "openai",
                api_key: "key-b",
                owner_id: OWNER_B,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stats_b.misses, 1,
            "owner B must not hit owner A's cache entry"
        );
    }

    #[tokio::test]
    async fn store_lookup_failure_aborts_before_any_provider_call() {
        let state = state_with(FailingStore);
        let provider = PanicProvider;
        let texts = vec!["text".to_string()];

        let result = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[tokio::test]
    async fn provider_failure_surfaces_as_app_error() {
        let state = state_with(MockStore::empty());
        let provider = FailingProvider;
        let texts = vec!["text".to_string()];

        let result = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Provider(_))));
    }

    #[tokio::test]
    async fn metrics_are_labeled_by_provider_and_only_recorded_on_success() {
        let state = state_with(MockStore::empty());
        let openai_provider = MockProvider {
            response: vec![vec![1.0]],
        };
        let texts = vec!["text".to_string()];

        embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(openai_provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        let mistral_texts = vec!["other text".to_string()];
        let failing_provider = FailingProvider;
        let _ = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(failing_provider),
                provider_name: "mistral",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &mistral_texts,
            },
        )
        .await;

        let metrics_text = state.metrics.encode();
        assert!(metrics_text
            .contains("zerocache_cache_misses_total{content_type=\"text\",provider=\"openai\"} 1"));
        assert!(
            !metrics_text.contains(
                "zerocache_cache_misses_total{content_type=\"text\",provider=\"mistral\"}"
            ),
            "a failed request must not record any metric for that provider"
        );
    }

    #[tokio::test]
    async fn different_provider_versions_produce_different_cache_entries() {
        struct VersionedProvider {
            version: &'static str,
            response: Vec<Vec<f32>>,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for VersionedProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                assert_eq!(texts.len(), self.response.len());
                Ok((self.response.clone(), ProviderUsage::default()))
            }

            fn version(&self) -> &'static str {
                self.version
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        let state = state_with(MockStore::empty());
        let texts = vec!["same text".to_string()];

        let v1 = VersionedProvider {
            version: "1.0.0",
            response: vec![vec![1.0]],
        };
        let (_, stats_v1) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(v1),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_v1.misses, 1);

        let v2 = VersionedProvider {
            version: "2.0.0",
            response: vec![vec![2.0]],
        };
        let (_, stats_v2) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(v2),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stats_v2.misses, 1,
            "a different adapter version must not hit the old version's cache entry"
        );
    }

    /// A provider mock that records every batch it is asked to embed and
    /// returns one distinct vector per call position, so a test can assert
    /// both *how many* provider calls happened and *what exact text* each
    /// one received.
    struct RecordingProvider {
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl RecordingProvider {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }

        fn recorded_calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for RecordingProvider {
        async fn embed_batch(
            &self,
            _api_key: &str,
            _model: &str,
            texts: &[String],
        ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
            self.calls.lock().unwrap().push(texts.to_vec());
            let vectors = (0..texts.len()).map(|i| vec![i as f32]).collect();
            Ok((vectors, ProviderUsage::default()))
        }

        fn version(&self) -> &'static str {
            "mock-v1"
        }

        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok("mock-scope".to_string())
        }
    }

    #[tokio::test]
    async fn casing_and_punctuation_variants_share_one_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider: Arc<RecordingProvider> = Arc::new(RecordingProvider::new());

        let first = vec!["Hello   World.".to_string()];
        let (_, stats_first) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &first,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats_first.misses, 1, "first phrasing must be a cold miss");

        let second = vec!["hello world".to_string()];
        let (_, stats_second) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &second,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            stats_second.hits, 1,
            "a casing/punctuation-only variant must hit the entry the first phrasing wrote"
        );
        assert_eq!(stats_second.misses, 0);

        let calls = provider.recorded_calls();
        assert_eq!(
            calls.len(),
            1,
            "the variant must not trigger a second provider call"
        );
        assert_eq!(
            calls[0],
            vec!["Hello World.".to_string()],
            "the provider must receive the whitespace-normalized original text, \
             not the lowercased/punctuation-stripped canonical form"
        );
    }

    #[tokio::test]
    async fn casing_variants_within_one_batch_call_provider_once() {
        let state = state_with(MockStore::empty());
        let provider: Arc<RecordingProvider> = Arc::new(RecordingProvider::new());

        let texts = vec![
            "Foo Bar".to_string(),
            "foo   bar".to_string(),
            "FOO BAR.".to_string(),
        ];
        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(stats.misses, 3, "all three positions are misses");
        let calls = provider.recorded_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec!["Foo Bar".to_string()],
            "the three canonical-equal variants collapse to one provider input, \
             the first position's whitespace-normalized form"
        );
        assert_eq!(
            vectors,
            vec![vec![0.0], vec![0.0], vec![0.0]],
            "every position gets the one fetched vector, order preserved"
        );
    }

    #[tokio::test]
    async fn delete_matches_entries_canonically() {
        let state = state_with(MockStore::empty());
        let provider: Arc<RecordingProvider> = Arc::new(RecordingProvider::new());

        let written = vec!["Hello World!".to_string()];
        embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &written,
            },
        )
        .await
        .unwrap();

        let to_delete = vec!["hello world".to_string()];
        let deleted = delete_batch(
            &state,
            DeleteRequest {
                provider: &*provider,
                provider_name: "openai",
                owner_id: OWNER_A,
                model: "m",
                texts: &to_delete,
            },
        )
        .await
        .unwrap();
        assert_eq!(deleted, 1);

        let (_, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &written,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stats.misses, 1,
            "a canonical-equal DELETE must have removed the entry the original POST wrote"
        );
    }

    #[tokio::test]
    async fn duplicate_texts_in_one_batch_call_provider_only_once() {
        struct CountingProvider {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for CountingProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                self.call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(
                    texts.len(),
                    1,
                    "duplicate texts in the batch must be deduplicated before calling the provider"
                );
                Ok((vec![vec![42.0]], ProviderUsage::default()))
            }

            fn version(&self) -> &'static str {
                "mock-v1"
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        let state = state_with(MockStore::empty());
        // Kept as a concrete Arc<CountingProvider>, not Arc<dyn
        // EmbeddingProvider>, so call_count is still inspectable after the
        // request -- EmbedRequest.provider gets its own coerced clone.
        let provider = Arc::new(CountingProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        // Same text three times in one batch.
        let texts = vec![
            "same text".to_string(),
            "same text".to_string(),
            "same text".to_string(),
        ];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            vectors,
            vec![vec![42.0], vec![42.0], vec![42.0]],
            "the same vector must be broadcast to every duplicate position"
        );
        assert_eq!(
            stats.misses, 3,
            "hit/miss accounting still reflects all three original positions"
        );
    }

    #[tokio::test]
    async fn concurrent_identical_misses_across_separate_requests_are_coalesced_into_one_provider_call(
    ) {
        // Distinct from duplicate_texts_in_one_batch_call_provider_only_once
        // above: that test dedupes duplicates *within* one batch. This one
        // proves the harder case -- request coalescing *across* separate,
        // genuinely concurrent embed_batch calls, which is what
        // fetch_coalesced/AppState.in_flight exists for. This is the exact
        // gap the LangChain TS battle-test measured live: 5 concurrent
        // requests for one never-before-seen text produced 5 real provider
        // calls, not 1.
        struct SlowCountingProvider {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for SlowCountingProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                self.call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Long enough that all 5 concurrent callers below genuinely
                // overlap in time rather than happening to run one after
                // another -- without this, the test could pass even if
                // coalescing were broken, just by getting lucky on timing.
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok((
                    vec![vec![99.0]; texts.len()],
                    ProviderUsage {
                        prompt_tokens: 7,
                        total_tokens: 7,
                    },
                ))
            }

            fn version(&self) -> &'static str {
                "mock-v1"
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        let state = state_with(MockStore::empty());
        // Concrete Arc<SlowCountingProvider>, same reasoning as
        // CountingProvider above: call_count needs to stay inspectable
        // after the requests complete.
        let provider = Arc::new(SlowCountingProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        let texts = vec!["never before seen concurrent text".to_string()];

        let requests = (0..5).map(|_| {
            embed_batch(
                &state,
                EmbedRequest {
                    provider: Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                    provider_name: "openai",
                    api_key: "key",
                    owner_id: OWNER_A,
                    model: "m",
                    texts: &texts,
                },
            )
        });

        let results = futures::future::join_all(requests).await;

        let mut total_prompt_tokens = 0;
        for result in &results {
            let (vectors, stats) = result
                .as_ref()
                .expect("every concurrent request must still succeed");
            assert_eq!(vectors, &vec![vec![99.0]], "every request must resolve to the correct vector, whether it claimed the fetch or piggybacked");
            total_prompt_tokens += stats.provider_prompt_tokens;
        }

        assert_eq!(
            provider.call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "5 concurrent requests missing on the exact same never-before-seen key must coalesce into exactly one provider call, not 5"
        );
        // Only the one request that actually claimed the fetch attributes
        // real usage to itself -- the other 4 piggybacked and correctly
        // contribute zero, exactly like a cache hit would. Order-
        // independent on purpose: which of the 5 wins the claim race isn't
        // deterministic.
        assert_eq!(total_prompt_tokens, 7, "usage must be attributed exactly once across all 5 requests, not once per request (double counting) or zero (lost)");
    }

    #[tokio::test]
    async fn whitespace_variants_of_the_same_text_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider = MockProvider {
            response: vec![vec![1.0]],
        };
        let texts = vec!["  hello   world  ".to_string()];

        embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        // A second request with different incidental whitespace, but the
        // same normalized content, must hit the entry the first request
        // wrote -- proving the store key was derived from normalized text.
        let panic_provider = PanicProvider;
        let differently_spaced_texts = vec!["hello world".to_string()];
        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(panic_provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &differently_spaced_texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(stats.hits, 1);
        assert_eq!(vectors, vec![vec![1.0]]);
    }

    #[tokio::test]
    async fn delete_batch_removes_entries_so_a_later_request_misses_again() {
        let key = CacheKey::derive(
            OWNER_A,
            "openai",
            "mock-scope",
            "m",
            "mock-v1",
            "to be forgotten",
        );
        let state = state_with(MockStore::with(vec![(key, vec![9.0])]));
        let provider: Arc<dyn EmbeddingProvider> = Arc::new(MockProvider {
            response: vec![vec![1.0]],
        });
        let texts = vec!["to be forgotten".to_string()];

        let deleted = delete_batch(
            &state,
            DeleteRequest {
                provider: provider.as_ref(),
                provider_name: "openai",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(deleted, 1);

        let (_, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&provider),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stats.hits, 0,
            "a deleted entry must be a miss again, not still cached"
        );
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn delete_batch_failure_surfaces_as_app_error() {
        let state = state_with(FailingStore);
        let provider = PanicProvider;
        let texts = vec!["text".to_string()];

        let result = delete_batch(
            &state,
            DeleteRequest {
                provider: &provider,
                provider_name: "openai",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await;

        assert!(matches!(result, Err(AppError::Store(_))));
    }

    #[tokio::test]
    async fn readiness_check_succeeds_against_a_healthy_store() {
        let state = state_with(MockStore::empty());
        assert!(check_store_readiness(&state).await.is_ok());
    }

    #[tokio::test]
    async fn readiness_check_fails_when_the_store_is_unreachable() {
        let state = state_with(FailingStore);
        assert!(matches!(
            check_store_readiness(&state).await,
            Err(AppError::Store(_))
        ));
    }

    #[tokio::test]
    async fn run_store_task_times_out_on_a_hung_closure() {
        let result: Result<(), AppError> =
            run_store_task_with_timeout(Duration::from_millis(20), || {
                std::thread::sleep(Duration::from_secs(5));
                Ok(())
            })
            .await;

        assert!(
            matches!(result, Err(AppError::Store(_))),
            "a closure that outlives the timeout must surface as a store timeout error"
        );
    }

    #[tokio::test]
    #[should_panic(expected = "store task panicked")]
    async fn run_store_task_propagates_a_panic_rather_than_reporting_it_as_a_timeout() {
        // A generous timeout, so if this test ever fails by reporting a
        // timeout instead of panicking, that's a real regression -- the
        // panic resolves almost instantly, long before the timeout could
        // fire on its own.
        let _: Result<(), AppError> = run_store_task_with_timeout(Duration::from_secs(30), || {
            panic!("simulated store task panic");
        })
        .await;
    }

    #[tokio::test]
    async fn image_all_misses_calls_provider_and_writes_back_to_store() {
        let state = state_with(MockStore::empty());
        let provider = Arc::new(MockImageProvider::returning(Ok((
            vec![vec![1.0], vec![2.0]],
            ProviderUsage::default(),
        ))));
        let images = vec![
            ImageInput {
                mime_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
            ImageInput {
                mime_type: "image/png".to_string(),
                data: "d29ybGQ=".to_string(),
            },
        ];

        let request = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };

        let (vectors, stats) = embed_image_batch(&state, request).await.unwrap();
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 2);
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn duplicate_images_in_one_batch_call_provider_only_once() {
        struct CountingImageProvider {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ImageEmbeddingProvider for CountingImageProvider {
            async fn embed_image_batch(
                &self,
                _api_key: &str,
                _model: &str,
                images: &[ImageInput],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                self.call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(images.len(), 1, "duplicate images in the batch must be deduplicated before calling the provider");
                Ok((vec![vec![42.0]], ProviderUsage::default()))
            }

            fn version(&self) -> &'static str {
                "test-image-v1"
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        let state = state_with(MockStore::empty());
        let provider = Arc::new(CountingImageProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        // Same image (identical bytes + mime type) three times in one batch.
        let images = vec![
            ImageInput {
                mime_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
            ImageInput {
                mime_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
            ImageInput {
                mime_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
        ];

        let request = EmbedImageRequest {
            provider: Arc::clone(&provider) as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };

        let (vectors, stats) = embed_image_batch(&state, request).await.unwrap();

        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            vectors,
            vec![vec![42.0], vec![42.0], vec![42.0]],
            "the same vector must be broadcast to every duplicate position"
        );
        assert_eq!(
            stats.misses, 3,
            "hit/miss accounting still reflects all three original positions"
        );
    }

    #[tokio::test]
    async fn concurrent_identical_image_misses_across_separate_requests_are_coalesced_into_one_provider_call(
    ) {
        // The image counterpart to
        // concurrent_identical_misses_across_separate_requests_are_coalesced_into_one_provider_call
        // above -- proves cross-request coalescing now also applies to
        // embed_image_batch, not just embed_batch (see CLAUDE.md Deviations
        // item 20).
        struct SlowCountingImageProvider {
            call_count: std::sync::atomic::AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ImageEmbeddingProvider for SlowCountingImageProvider {
            async fn embed_image_batch(
                &self,
                _api_key: &str,
                _model: &str,
                images: &[ImageInput],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                self.call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Long enough that all 5 concurrent callers below genuinely
                // overlap in time rather than happening to run one after
                // another -- without this, the test could pass even if
                // coalescing were broken, just by getting lucky on timing.
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok((
                    vec![vec![99.0]; images.len()],
                    ProviderUsage {
                        prompt_tokens: 7,
                        total_tokens: 7,
                    },
                ))
            }

            fn version(&self) -> &'static str {
                "mock-image-v1"
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        let state = state_with(MockStore::empty());
        let provider = Arc::new(SlowCountingImageProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        let images = vec![ImageInput {
            mime_type: "image/png".to_string(),
            data: "bmV2ZXItYmVmb3JlLXNlZW4=".to_string(),
        }];

        let requests = (0..5).map(|_| {
            embed_image_batch(
                &state,
                EmbedImageRequest {
                    provider: Arc::clone(&provider) as Arc<dyn ImageEmbeddingProvider>,
                    provider_name: "gemini",
                    api_key: "test-key",
                    owner_id: [1u8; 32],
                    model: "gemini-embedding-2",
                    images: &images,
                },
            )
        });

        let results = futures::future::join_all(requests).await;

        let mut total_prompt_tokens = 0;
        for result in &results {
            let (vectors, stats) = result
                .as_ref()
                .expect("every concurrent request must still succeed");
            assert_eq!(vectors, &vec![vec![99.0]], "every request must resolve to the correct vector, whether it claimed the fetch or piggybacked");
            total_prompt_tokens += stats.provider_prompt_tokens;
        }

        assert_eq!(
            provider.call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "5 concurrent requests missing on the exact same never-before-seen image key must coalesce into exactly one provider call, not 5"
        );
        assert_eq!(total_prompt_tokens, 7, "usage must be attributed exactly once across all 5 requests, not once per request (double counting) or zero (lost)");
    }

    #[tokio::test]
    async fn image_all_hits_skips_provider_entirely() {
        let key = CacheKey::derive_image(
            [1u8; 32],
            "gemini",
            "mock-scope",
            "gemini-embedding-2",
            "test-image-v1",
            "image/png",
            "aGVsbG8=",
        );
        let state = state_with(MockStore::with(vec![(key, vec![9.0])]));

        let provider = Arc::new(MockImageProvider::returning(Ok((
            vec![],
            ProviderUsage::default(),
        ))));
        let images = vec![ImageInput {
            mime_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        }];

        let request = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };

        let (vectors, stats) = embed_image_batch(&state, request).await.unwrap();
        assert_eq!(vectors, vec![vec![9.0]]);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn image_different_owners_never_share_a_cache_entry() {
        let state = state_with(MockStore::empty());
        let provider = Arc::new(MockImageProvider::returning(Ok((
            vec![vec![1.0]],
            ProviderUsage::default(),
        ))));
        let images = vec![ImageInput {
            mime_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        }];

        let request_a = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };
        embed_image_batch(&state, request_a).await.unwrap();

        let request_b = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [2u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };
        let (_vectors, stats_b) = embed_image_batch(&state, request_b).await.unwrap();

        assert_eq!(
            stats_b.misses, 1,
            "a different owner_id must never hit the first owner's cache entry"
        );
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[tokio::test]
    async fn delete_image_batch_removes_entries_so_a_later_request_misses_again() {
        let state = state_with(MockStore::empty());
        let provider = Arc::new(MockImageProvider::returning(Ok((
            vec![vec![1.0]],
            ProviderUsage::default(),
        ))));
        let images = vec![ImageInput {
            mime_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        }];

        let embed_request = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };
        embed_image_batch(&state, embed_request).await.unwrap();
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        let delete_request = DeleteImageRequest {
            provider: provider.as_ref(),
            provider_name: "gemini",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };
        let deleted = delete_image_batch(&state, delete_request).await.unwrap();
        assert_eq!(deleted, 1);

        let embed_request_again = EmbedImageRequest {
            provider: provider.clone() as Arc<dyn ImageEmbeddingProvider>,
            provider_name: "gemini",
            api_key: "test-key",
            owner_id: [1u8; 32],
            model: "gemini-embedding-2",
            images: &images,
        };
        embed_image_batch(&state, embed_request_again)
            .await
            .unwrap();
        assert_eq!(
            provider
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "after deletion, the same image must miss again and call the provider a second time"
        );
    }

    #[tokio::test]
    async fn two_providers_with_different_cache_scopes_never_share_a_cache_entry() {
        // The application-level statement of what zerocache-core proves in
        // isolation: same owner, same provider name, same model, same text --
        // but two adapter instances pointed at different upstreams. The
        // second must miss, not inherit the first's vector.
        struct ScopedProvider {
            scope: &'static str,
            response: Vec<Vec<f32>>,
        }

        #[async_trait::async_trait]
        impl EmbeddingProvider for ScopedProvider {
            async fn embed_batch(
                &self,
                _api_key: &str,
                _model: &str,
                _texts: &[String],
            ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
                Ok((
                    self.response.clone(),
                    ProviderUsage {
                        prompt_tokens: 1,
                        total_tokens: 1,
                    },
                ))
            }

            fn version(&self) -> &'static str {
                "mock-v1"
            }

            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok(self.scope.to_string())
            }
        }

        let state = state_with(MockStore::empty());
        let texts = vec!["identical text".to_string()];

        let first: Arc<dyn EmbeddingProvider> = Arc::new(ScopedProvider {
            scope: "https://api.openai.com",
            response: vec![vec![1.0]],
        });
        let (_, stats_first) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&first),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stats_first.misses, 1,
            "first call against a cold store must miss"
        );

        let second: Arc<dyn EmbeddingProvider> = Arc::new(ScopedProvider {
            scope: "http://localhost:8000",
            response: vec![vec![2.0]],
        });
        let (vectors, stats_second) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::clone(&second),
                provider_name: "openai",
                api_key: "key",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            stats_second.misses, 1,
            "a different upstream endpoint must not hit the first one's entry"
        );
        assert_eq!(
            vectors,
            vec![vec![2.0]],
            "and must return its own vector, not the cached one"
        );
    }

    // --- cross-replica single-flight for lone-key embedding misses ---

    use zerocache_ports::{CoalescingCoordinator, FollowSignal, Role};

    /// Always a follower whose `follow` returns immediately -- models a peer
    /// replica that has already claimed and is filling this key.
    struct FollowerCoord;
    impl CoalescingCoordinator for FollowerCoord {
        fn try_lead(&self, _k: &CacheKey) -> Role {
            Role::Follower
        }
        fn complete(&self, _k: &CacheKey) {}
        fn follow(&self, _k: &CacheKey, _w: std::time::Duration) -> FollowSignal {
            FollowSignal::Signalled
        }
    }

    /// Panics on `try_lead` -- proves a multi-key claim never reaches the
    /// coordinator.
    struct AssertNeverContendedCoord;
    impl CoalescingCoordinator for AssertNeverContendedCoord {
        fn try_lead(&self, _k: &CacheKey) -> Role {
            panic!("multi-key batch must not enter the cross-replica path");
        }
        fn complete(&self, _k: &CacheKey) {}
        fn follow(&self, _k: &CacheKey, _w: std::time::Duration) -> FollowSignal {
            FollowSignal::WaitElapsed
        }
    }

    /// `get` misses the first call (so `reconcile` records a miss) then hits
    /// every later call (so the coalesced re-read finds a peer's fill).
    struct MissThenHitStore {
        value: Vec<f32>,
        gets: std::sync::atomic::AtomicUsize,
    }
    impl EmbeddingStore for MissThenHitStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            let n = self.gets.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(if n == 0 {
                None
            } else {
                Some(self.value.clone())
            })
        }
        fn put(&self, _key: CacheKey, _v: Vec<f32>) -> Result<(), StoreError> {
            Ok(())
        }
        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_single_key_miss_served_by_a_peer_fill_makes_no_provider_call() {
        let mut state = state_with(MissThenHitStore {
            value: vec![3.0, 3.0],
            gets: std::sync::atomic::AtomicUsize::new(0),
        });
        state.coordinator = Arc::new(FollowerCoord);
        let texts = vec!["solo".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(PanicProvider), // must never be called
                provider_name: "openai",
                api_key: "k",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();

        assert_eq!(vectors, vec![vec![3.0, 3.0]]);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 0, "a peer fill bills nothing");

        let dump = state.metrics.encode();
        assert!(
            dump.contains(
                "zerocache_cross_replica_coalesced_total{kind=\"embedding\",provider=\"openai\"} 1"
            ),
            "{dump}"
        );
    }

    #[tokio::test]
    async fn a_multi_key_batch_never_enters_the_cross_replica_path() {
        let mut state = state_with(MockStore::empty());
        state.coordinator = Arc::new(AssertNeverContendedCoord);
        let provider = MockProvider {
            response: vec![vec![1.0], vec![2.0]],
        };
        let texts = vec!["alpha".to_string(), "beta".to_string()];

        let (_v, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "k",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(stats.misses, 2); // AssertNeverContendedCoord.try_lead would panic
    }

    #[tokio::test]
    async fn noop_coordinator_single_key_miss_still_calls_the_provider_and_writes_back() {
        let state = state_with(MockStore::empty()); // default NoopCoordinator
        let provider = MockProvider {
            response: vec![vec![9.0]],
        };
        let texts = vec!["fresh".to_string()];

        let (vectors, stats) = embed_batch(
            &state,
            EmbedRequest {
                provider: Arc::new(provider),
                provider_name: "openai",
                api_key: "k",
                owner_id: OWNER_A,
                model: "m",
                texts: &texts,
            },
        )
        .await
        .unwrap();
        assert_eq!(vectors, vec![vec![9.0]]);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.provider_prompt_tokens, 10);
        let key = CacheKey::derive(OWNER_A, "openai", "mock-scope", "m", "mock-v1", "fresh");
        assert_eq!(state.store.get(&key).unwrap(), Some(vec![9.0]));
    }
}

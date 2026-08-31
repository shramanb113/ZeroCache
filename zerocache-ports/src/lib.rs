use std::fmt;

use zerocache_core::CacheKey;

#[derive(Debug)]
pub struct StoreError(pub String);

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "store error: {}", self.0)
    }
}

impl std::error::Error for StoreError {}

// Clone is needed so a ProviderError can flow through a coalesced,
// futures::future::Shared in-flight fetch (zerocache-http's request
// coalescing) -- every waiter on a shared future gets its own clone of the
// resolved Result, error included.
#[derive(Debug, Clone)]
pub struct ProviderError(pub String);

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provider error: {}", self.0)
    }
}

impl std::error::Error for ProviderError {}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

pub trait EmbeddingStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError>;
    fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError>;
    fn delete(&self, key: &CacheKey) -> Result<(), StoreError>;
}

/// Opaque byte store for cached chat-completion responses. Separate from
/// `EmbeddingStore` because the stored value is a serialized response
/// record, not a raw f32 vector -- a `sled`/`redis` adapter implements both
/// traits on one struct. The record format (response body + token counts)
/// is `zerocache-http`'s concern; this trait only moves bytes.
pub trait CompletionStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<u8>>, StoreError>;
    fn put(&self, key: CacheKey, value: Vec<u8>) -> Result<(), StoreError>;
    fn delete(&self, key: &CacheKey) -> Result<(), StoreError>;
}

/// A persisted semantic-index entry: an L2-normalized embedding of a request's
/// fuzzy span plus what's needed to rebuild the in-memory index and find the
/// completion. `index_version` lets a reader skip records from an older format.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub exact_key: CacheKey,
    pub scope_hash: [u8; 32],
    pub coarse_key_hash: [u8; 32],
    pub index_version: u8,
    pub vector: Vec<f32>,
}

/// An incremental slice of index changes since a cursor, plus the cursor to
/// pass on the next call. `changes_since(None)` means "everything from the
/// start" and is how a replica does its initial load on the redis backend.
/// A backend with no shared feed (sled) returns `Default` -- empty,
/// `cursor: None` -- and the caller then relies on `load_all`.
#[derive(Debug, Clone, Default)]
pub struct VectorChanges {
    pub upserts: Vec<VectorRecord>,
    /// `(exact_key, scope_hash)` -- the scope so the caller can tombstone the
    /// right per-scope graph in O(1) without scanning every scope.
    pub deletes: Vec<(CacheKey, [u8; 32])>,
    /// Opaque resume token (a Redis stream ID for the redis impl). `None`
    /// means no change-feed: do not spawn a poll loop, use `load_all`.
    pub cursor: Option<String>,
}

/// Persistence for `VectorRecord`s. Separate from `CompletionStore` because it
/// must enumerate (`load_all`) to rebuild the index at boot. sled only in v1.
pub trait CompletionVectorStore: Send + Sync {
    fn insert(&self, record: VectorRecord) -> Result<(), StoreError>;
    fn delete(&self, exact_key: &CacheKey, scope_hash: &[u8; 32]) -> Result<(), StoreError>;
    fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError>;

    /// Changes since `cursor` (exclusive). `None` = full replay. Returns
    /// `cursor: None` on a backend with no shared feed (sled), signalling the
    /// caller to skip the poll loop and rely on `load_all`.
    fn changes_since(&self, cursor: Option<String>) -> Result<VectorChanges, StoreError>;

    /// Like `changes_since(Some(cursor))`, but a backend with a native blocking
    /// read (redis `XREAD BLOCK`) waits up to `timeout` for the first entry
    /// after `cursor` before returning, cutting change-feed propagation lag to
    /// roughly one round-trip. On timeout with nothing new it returns empty
    /// `upserts`/`deletes` and `cursor: Some(<unchanged cursor>)`. A backend
    /// with no shared feed (sled) has no such primitive and returns
    /// `VectorChanges::default()` immediately -- it never runs the poll loop
    /// that calls this.
    fn changes_blocking(
        &self,
        cursor: String,
        timeout: std::time::Duration,
    ) -> Result<VectorChanges, StoreError>;
}

/// Which side of a cross-replica single-flight this replica is on for one key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Perform the provider call, then call `complete`.
    Leader,
    /// A peer holds the lock; call `follow` and re-read the store.
    Follower,
}

/// Result of one bounded `follow` wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FollowSignal {
    /// The leader signalled (fill done or failed) -- re-read the store now.
    Signalled,
    /// `wait` elapsed with no signal.
    WaitElapsed,
}

/// Distributed single-flight for one `CacheKey`. Synchronous: callers run it
/// on a blocking thread, like the store traits. Every method degrades to a
/// safe default on any internal (e.g. Redis) error -- a coordination outage
/// falls back to per-replica behaviour, never a failed request.
pub trait CoalescingCoordinator: Send + Sync {
    /// `Leader` => this replica fetches and must call `complete` after.
    /// `Follower` => await `follow`. Any error => `Leader`.
    fn try_lead(&self, key: &CacheKey) -> Role;
    /// Leader only: release the lock and wake followers. Call on success and
    /// on failure. Errors are ignored (the lock TTLs out).
    fn complete(&self, key: &CacheKey);
    /// Follower only: block up to `wait` for the leader's signal. Any error
    /// => `WaitElapsed`.
    fn follow(&self, key: &CacheKey, wait: std::time::Duration) -> FollowSignal;
}

/// Token counts from a chat-completion response, used only for the
/// tokens-saved metric on a cache hit. All zero when the provider omits a
/// usage block or the upstream call returned a non-2xx.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// One upstream chat-completion or Anthropic-message result. `status` is the HTTP status the
/// provider returned: the proxy forwards a non-2xx body to the caller
/// verbatim and never caches it, so the adapter surfaces it as `Ok` with
/// the real status rather than `Err`. `Err(ProviderError)` is reserved for
/// transport failures (connection, timeout) where there is no response at
/// all.
///
/// `Clone` so a resolved response can flow through a coalesced in-flight
/// fetch, the same way `EmbeddingProvider`'s results already do.
#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub status: u16,
    pub body: serde_json::Value,
    pub usage: CompletionUsage,
}

#[async_trait::async_trait]
pub trait ChatCompletionProvider: Send + Sync {
    /// Forwards `request` -- a full OpenAI `/v1/chat/completions` JSON body
    /// -- to the upstream endpoint using the caller's `api_key`, returning
    /// the response as-is (no wire-shape translation; the chat shape is the
    /// contract on both sides).
    async fn chat_completion(
        &self,
        api_key: &str,
        request: &serde_json::Value,
    ) -> Result<ChatCompletionResponse, ProviderError>;

    /// Adapter build identifier for cache-key versioning -- see
    /// `EmbeddingProvider::version`.
    fn version(&self) -> &'static str;

    /// Upstream-weights identifier for `model` -- see
    /// `EmbeddingProvider::cache_scope`. For the OpenAI-shaped chat adapter
    /// this is just the configured base URL.
    fn cache_scope(&self, model: &str) -> Result<String, ProviderError>;
}

/// A stream of raw upstream SSE bytes. Items are arbitrary byte chunks whose
/// boundaries do NOT align to SSE frame boundaries -- the caller
/// (`zerocache-http/src/sse.rs`) owns framing and delta assembly. An `Err`
/// item is a mid-stream transport failure.
pub type SseByteStream =
    std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<Vec<u8>, ProviderError>> + Send>>;

/// Streaming counterpart to `ChatCompletionProvider`. A separate trait (like
/// `ImageEmbeddingProvider`) so the registry and wiring stay explicit;
/// `OpenAiWireChatProvider` implements both.
#[async_trait::async_trait]
pub trait StreamingChatCompletionProvider: Send + Sync {
    /// Opens an upstream streaming chat completion with the caller's key.
    /// `Err` = transport failure before any bytes. `Ok((status, stream))`
    /// carries the upstream HTTP status; a non-2xx status means the stream
    /// yields the error body, not SSE frames.
    async fn chat_completion_stream(
        &self,
        api_key: &str,
        request: &serde_json::Value,
    ) -> Result<(u16, SseByteStream), ProviderError>;

    /// See `ChatCompletionProvider::version`.
    fn version(&self) -> &'static str;

    /// See `ChatCompletionProvider::cache_scope`.
    fn cache_scope(&self, model: &str) -> Result<String, ProviderError>;
}

/// Extra request headers the Messages proxy forwards to Anthropic verbatim.
/// `anthropic_beta` is additionally folded into the cache key by the
/// orchestrator (`zerocache-http/src/messages.rs`) — a beta feature can
/// change the response shape.
#[derive(Debug, Clone, Default)]
pub struct MessageHeaders {
    pub anthropic_version: Option<String>,
    pub anthropic_beta: Option<String>,
}

/// Anthropic native `/v1/messages` proxy. A separate trait from
/// `ChatCompletionProvider` (not an extension) because the wire shape is
/// different on both sides — Anthropic's request/response JSON and its
/// `x-api-key` auth — and callers using the Anthropic SDK expect their exact
/// bytes back. Reuses `ChatCompletionResponse` (structurally wire-neutral)
/// with the adapter mapping `input_tokens` -> `prompt_tokens`, `output_tokens`
/// -> `completion_tokens`, and their sum -> `total_tokens`.
#[async_trait::async_trait]
pub trait MessagesProvider: Send + Sync {
    /// Forwards `request` (a full Anthropic `/v1/messages` body) upstream with
    /// the caller's `api_key` as `x-api-key`. Non-2xx -> `Ok` with the real
    /// status (forwarded verbatim, never cached); `Err` = transport failure.
    async fn messages(
        &self,
        api_key: &str,
        request: &serde_json::Value,
        headers: &MessageHeaders,
    ) -> Result<ChatCompletionResponse, ProviderError>;

    /// Raw byte pipe for `stream: true` — no framing, no store, no metrics.
    /// `Err` = transport failure before any bytes. `Ok((status, stream))`
    /// carries the upstream HTTP status; a non-2xx status means the stream
    /// yields the error body, not SSE frames. Buffer-and-replay for
    /// `/v1/messages` is deferred, so this is the whole streaming story here.
    async fn messages_stream_passthrough(
        &self,
        api_key: &str,
        request: &serde_json::Value,
        headers: &MessageHeaders,
    ) -> Result<(u16, SseByteStream), ProviderError>;

    /// See `ChatCompletionProvider::version`.
    fn version(&self) -> &'static str;

    /// See `ChatCompletionProvider::cache_scope`. For the Anthropic adapter
    /// this is just the configured base URL.
    fn cache_scope(&self, model: &str) -> Result<String, ProviderError>;
}

#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError>;

    /// Identifies this adapter's own build for cache-key purposes — tied to
    /// the adapter crate's `Cargo.toml` version, not a manually maintained
    /// string, so a behavior change is invisible in the cache key only if
    /// the crate version wasn't bumped, the same discipline every published
    /// crate already needs.
    fn version(&self) -> &'static str;

    /// A string that fully identifies *which upstream weights* would answer a
    /// request for `model` through this adapter instance. Folded into the
    /// cache key alongside owner/provider/model/version.
    ///
    /// Two adapter configurations that could return different vectors for the
    /// same (owner, provider, model, version, text) tuple MUST return
    /// different strings here -- this direction is a hard requirement, since
    /// getting it wrong risks serving a wrong vector as if it were correct.
    /// The reverse is only a *should*, not a MUST: two configurations that
    /// are guaranteed to return the same vector should return the same
    /// string, but returning different strings for them is always safe (it
    /// just costs an unnecessary cold miss, never a wrong answer), so a more
    /// conservative implementation is free to over-distinguish. For the four
    /// wire-shape-fixed adapters this is just the configured base URL --
    /// repointing ZEROCACHE_OPENAI_BASE_URL at a self-hosted vLLM must not
    /// silently reuse vectors computed by api.openai.com. For a cloud adapter
    /// it also carries whatever per-request coordinates the caller encoded in
    /// `model` (region, project, deployment, task type), since those select
    /// different weights behind an identical model name -- and may also fold
    /// in coarser-grained implementation details (e.g. a shared client
    /// library's own version) that over-invalidate rather than risk
    /// under-invalidating.
    ///
    /// Fallible because a cloud adapter derives it by parsing `model`, which
    /// the caller can get wrong.
    fn cache_scope(&self, model: &str) -> Result<String, ProviderError>;
}

/// One image to embed: raw base64 payload plus the MIME type Gemini's
/// `inline_data` part needs to interpret it. The `data:...;base64,` prefix a
/// caller sends over HTTP is stripped before this struct is built -- that
/// parsing is wire-shape translation, so it lives in zerocache-http, not here.
///
/// Clone so zerocache-http's fetch_image_coalesced can cheaply hand a copy
/// into the 'static future backing a coalesced fetch, the same way `String`
/// texts are cloned for the text path's equivalent future.
#[derive(Clone)]
pub struct ImageInput {
    pub mime_type: String,
    pub data: String,
}

/// A separate trait from `EmbeddingProvider`, not an extension of it, because
/// only one of the four provider adapters can implement it for real (see
/// CLAUDE.md Deviations: OpenAI has no public image-embedding API at all) --
/// a default-returns-"unsupported" method on `EmbeddingProvider` itself would
/// force every future text-only adapter to carry dead code for a capability
/// it can never have.
#[async_trait::async_trait]
pub trait ImageEmbeddingProvider: Send + Sync {
    async fn embed_image_batch(
        &self,
        api_key: &str,
        model: &str,
        images: &[ImageInput],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError>;

    fn version(&self) -> &'static str;

    /// A string that fully identifies *which upstream weights* would answer a
    /// request for `model` through this adapter instance. Folded into the
    /// cache key alongside owner/provider/model/version.
    ///
    /// Two adapter configurations that could return different vectors for the
    /// same (owner, provider, model, version, text) tuple MUST return
    /// different strings here -- this direction is a hard requirement, since
    /// getting it wrong risks serving a wrong vector as if it were correct.
    /// The reverse is only a *should*, not a MUST: two configurations that
    /// are guaranteed to return the same vector should return the same
    /// string, but returning different strings for them is always safe (it
    /// just costs an unnecessary cold miss, never a wrong answer), so a more
    /// conservative implementation is free to over-distinguish. For the four
    /// wire-shape-fixed adapters this is just the configured base URL --
    /// repointing ZEROCACHE_OPENAI_BASE_URL at a self-hosted vLLM must not
    /// silently reuse vectors computed by api.openai.com. For a cloud adapter
    /// it also carries whatever per-request coordinates the caller encoded in
    /// `model` (region, project, deployment, task type), since those select
    /// different weights behind an identical model name -- and may also fold
    /// in coarser-grained implementation details (e.g. a shared client
    /// library's own version) that over-invalidate rather than risk
    /// under-invalidating.
    ///
    /// Fallible because a cloud adapter derives it by parsing `model`, which
    /// the caller can get wrong.
    fn cache_scope(&self, model: &str) -> Result<String, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mem(std::sync::Mutex<Vec<VectorRecord>>);

    impl CompletionVectorStore for Mem {
        fn insert(&self, record: VectorRecord) -> Result<(), StoreError> {
            self.0.lock().unwrap().push(record);
            Ok(())
        }
        fn delete(&self, exact_key: &CacheKey, _scope_hash: &[u8; 32]) -> Result<(), StoreError> {
            self.0.lock().unwrap().retain(|r| &r.exact_key != exact_key);
            Ok(())
        }
        fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn changes_since(&self, _cursor: Option<String>) -> Result<VectorChanges, StoreError> {
            Ok(VectorChanges::default())
        }
        fn changes_blocking(
            &self,
            _cursor: String,
            _timeout: std::time::Duration,
        ) -> Result<VectorChanges, StoreError> {
            Ok(VectorChanges::default())
        }
    }

    #[test]
    fn coalescing_coordinator_is_object_safe_and_noop_impl_always_leads() {
        struct AlwaysLeads;
        impl CoalescingCoordinator for AlwaysLeads {
            fn try_lead(&self, _key: &CacheKey) -> Role {
                Role::Leader
            }
            fn complete(&self, _key: &CacheKey) {}
            fn follow(&self, _key: &CacheKey, _wait: std::time::Duration) -> FollowSignal {
                FollowSignal::WaitElapsed
            }
        }

        let c: Box<dyn CoalescingCoordinator> = Box::new(AlwaysLeads);
        let key = CacheKey::from_bytes([5u8; 32]);
        assert_eq!(c.try_lead(&key), Role::Leader);
        assert_eq!(
            c.follow(&key, std::time::Duration::from_millis(1)),
            FollowSignal::WaitElapsed
        );
        c.complete(&key);
    }

    #[test]
    fn completion_vector_store_is_object_safe_and_round_trips() {
        let store: Box<dyn CompletionVectorStore> = Box::new(Mem(Default::default()));
        let key = CacheKey::from_bytes([7u8; 32]);
        store
            .insert(VectorRecord {
                exact_key: key,
                scope_hash: [1u8; 32],
                coarse_key_hash: [2u8; 32],
                index_version: 1,
                vector: vec![0.1, 0.2, 0.3],
            })
            .unwrap();
        assert_eq!(store.load_all().unwrap().len(), 1);
        store.delete(&key, &[1u8; 32]).unwrap();
        assert!(store.load_all().unwrap().is_empty());
    }

    #[test]
    fn changes_since_default_is_empty_with_no_cursor() {
        let store: Box<dyn CompletionVectorStore> = Box::new(Mem(Default::default()));
        let changes = store.changes_since(None).unwrap();
        assert!(changes.upserts.is_empty());
        assert!(changes.deletes.is_empty());
        assert!(changes.cursor.is_none());
    }

    #[test]
    fn changes_blocking_is_object_safe_and_empty_on_a_feedless_store() {
        let store: Box<dyn CompletionVectorStore> = Box::new(Mem(Default::default()));
        let changes = store
            .changes_blocking("0-0".to_string(), std::time::Duration::from_millis(1))
            .unwrap();
        assert!(changes.upserts.is_empty());
        assert!(changes.deletes.is_empty());
    }

    #[tokio::test]
    async fn streaming_chat_completion_provider_is_object_safe() {
        struct Fake;

        #[async_trait::async_trait]
        impl StreamingChatCompletionProvider for Fake {
            async fn chat_completion_stream(
                &self,
                _api_key: &str,
                _request: &serde_json::Value,
            ) -> Result<(u16, SseByteStream), ProviderError> {
                let frames: Vec<Result<Vec<u8>, ProviderError>> = vec![
                    Ok(b"data: {\"x\":1}\n\n".to_vec()),
                    Ok(b"data: [DONE]\n\n".to_vec()),
                ];
                let s = futures_util::stream::iter(frames);
                Ok((200, Box::pin(s)))
            }
            fn version(&self) -> &'static str {
                "fake-v1"
            }
            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("fake-scope".into())
            }
        }

        let p: Box<dyn StreamingChatCompletionProvider> = Box::new(Fake);
        let (status, mut stream) = p
            .chat_completion_stream("k", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(status, 200);
        let mut seen = 0;
        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            item.unwrap();
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn messages_provider_is_object_safe_and_headers_default_all_none() {
        struct Fake;

        #[async_trait::async_trait]
        impl MessagesProvider for Fake {
            async fn messages(
                &self,
                _api_key: &str,
                _request: &serde_json::Value,
                _headers: &MessageHeaders,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                Ok(ChatCompletionResponse {
                    status: 200,
                    body: serde_json::json!({"type": "message"}),
                    usage: CompletionUsage::default(),
                })
            }
            async fn messages_stream_passthrough(
                &self,
                _api_key: &str,
                _request: &serde_json::Value,
                _headers: &MessageHeaders,
            ) -> Result<(u16, SseByteStream), ProviderError> {
                let s = futures_util::stream::iter(vec![Ok(b"data: {}\n\n".to_vec())]);
                Ok((200, Box::pin(s)))
            }
            fn version(&self) -> &'static str {
                "fake-v1"
            }
            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("https://api.anthropic.com".into())
            }
        }

        let p: Box<dyn MessagesProvider> = Box::new(Fake);
        let resp = p
            .messages("k", &serde_json::json!({}), &MessageHeaders::default())
            .await
            .unwrap();
        assert_eq!(resp.status, 200);

        let h = MessageHeaders::default();
        assert!(h.anthropic_version.is_none());
        assert!(h.anthropic_beta.is_none());
    }
}

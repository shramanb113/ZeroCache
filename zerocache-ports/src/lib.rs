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

/// Token counts from a chat-completion response, used only for the
/// tokens-saved metric on a cache hit. All zero when the provider omits a
/// usage block or the upstream call returned a non-2xx.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// One upstream chat-completion result. `status` is the HTTP status the
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

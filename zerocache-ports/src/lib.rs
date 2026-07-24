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
}

use zerocache_core::CacheKey;

pub trait EmbeddingStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Option<Vec<f32>>;
    fn put(&self, key: CacheKey, vector: Vec<f32>);
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed_batch(&self, model: &str, texts: &[String]) -> Vec<Vec<f32>>;
}

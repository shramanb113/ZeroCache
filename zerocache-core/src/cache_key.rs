#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    // Fields are hashed with separators, not naive concatenation, so
    // e.g. ("gpt", "4-embed", "x") and ("gpt-4", "embed", "x") can't collide.
    pub fn derive(owner_id: [u8; 32], provider: &str, model: &str, model_version: &str, text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&owner_id);
        hasher.update(b"\0");
        hasher.update(provider.as_bytes());
        hasher.update(b"\0");
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(model_version.as_bytes());
        hasher.update(b"\0");
        hasher.update(text.as_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER_A: [u8; 32] = [1u8; 32];
    const OWNER_B: [u8; 32] = [2u8; 32];

    #[test]
    fn same_inputs_produce_same_key() {
        let a = CacheKey::derive(OWNER_A, "openai", "gpt-4-embed", "v1", "hello world");
        let b = CacheKey::derive(OWNER_A, "openai", "gpt-4-embed", "v1", "hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn different_model_version_produces_different_key() {
        let a = CacheKey::derive(OWNER_A, "openai", "gpt-4-embed", "v1", "hello world");
        let b = CacheKey::derive(OWNER_A, "openai", "gpt-4-embed", "v2", "hello world");
        assert_ne!(a, b);
    }

    #[test]
    fn field_boundary_is_not_ambiguous() {
        let a = CacheKey::derive(OWNER_A, "openai", "gpt", "4-embed", "x");
        let b = CacheKey::derive(OWNER_A, "openai", "gpt-4", "embed", "x");
        assert_ne!(a, b);
    }

    #[test]
    fn different_owner_produces_different_key() {
        let a = CacheKey::derive(OWNER_A, "openai", "gpt-4-embed", "v1", "same text");
        let b = CacheKey::derive(OWNER_B, "openai", "gpt-4-embed", "v1", "same text");
        assert_ne!(a, b, "two different callers must never share a cache entry");
    }

    #[test]
    fn different_provider_produces_different_key() {
        let a = CacheKey::derive(OWNER_A, "openai", "embed-v1", "v1", "same text");
        let b = CacheKey::derive(OWNER_A, "mistral", "embed-v1", "v1", "same text");
        assert_ne!(a, b, "two providers with an identically-named model must never collide");
    }
}

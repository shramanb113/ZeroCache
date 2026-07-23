#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    // Fields are hashed with separators, not naive concatenation, so
    // ("gpt", "4-embed", "x") and ("gpt-4", "embed", "x") can't collide.
    pub fn derive(model_id: &str, model_version: &str, text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_id.as_bytes());
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

    #[test]
    fn same_inputs_produce_same_key() {
        let a = CacheKey::derive("gpt-4-embed", "v1", "hello world");
        let b = CacheKey::derive("gpt-4-embed", "v1", "hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn different_model_version_produces_different_key() {
        let a = CacheKey::derive("gpt-4-embed", "v1", "hello world");
        let b = CacheKey::derive("gpt-4-embed", "v2", "hello world");
        assert_ne!(a, b);
    }

    #[test]
    fn field_boundary_is_not_ambiguous() {
        let a = CacheKey::derive("gpt", "4-embed", "x");
        let b = CacheKey::derive("gpt-4", "embed", "x");
        assert_ne!(a, b);
    }
}

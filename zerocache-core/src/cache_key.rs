#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    // Fields are hashed with separators, not naive concatenation, so
    // e.g. ("gpt", "4-embed", "x") and ("gpt-4", "embed", "x") can't collide.
    //
    // `cache_scope` identifies the concrete upstream endpoint the vector
    // would come from (see EmbeddingProvider::cache_scope). Without it, two
    // deployments pointed at different endpoints under the same provider name
    // and model name share a cache line, and one serves the other's vectors.
    pub fn derive(
        owner_id: [u8; 32],
        provider: &str,
        cache_scope: &str,
        model: &str,
        model_version: &str,
        text: &str,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&owner_id);
        hasher.update(b"\0");
        hasher.update(provider.as_bytes());
        hasher.update(b"\0");
        hasher.update(cache_scope.as_bytes());
        hasher.update(b"\0");
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(model_version.as_bytes());
        hasher.update(b"\0");
        hasher.update(text.as_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    /// Domain-separated from `derive` via the "image\0" literal below the
    /// model-version field -- guarantees an image-derived key can never
    /// collide with a text-derived key even if the image's base64 payload
    /// happens to equal some unrelated text byte-for-byte.
    pub fn derive_image(
        owner_id: [u8; 32],
        provider: &str,
        cache_scope: &str,
        model: &str,
        model_version: &str,
        mime_type: &str,
        data_base64: &str,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&owner_id);
        hasher.update(b"\0");
        hasher.update(provider.as_bytes());
        hasher.update(b"\0");
        hasher.update(cache_scope.as_bytes());
        hasher.update(b"\0");
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(model_version.as_bytes());
        hasher.update(b"\0");
        hasher.update(b"image\0");
        hasher.update(mime_type.as_bytes());
        hasher.update(b"\0");
        hasher.update(data_base64.as_bytes());
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
    const SCOPE_A: &str = "https://api.openai.com";
    const SCOPE_B: &str = "http://localhost:8000";

    #[test]
    fn same_inputs_produce_same_key() {
        let a = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_A,
            "gpt-4-embed",
            "v1",
            "hello world",
        );
        let b = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_A,
            "gpt-4-embed",
            "v1",
            "hello world",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn different_model_version_produces_different_key() {
        let a = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_A,
            "gpt-4-embed",
            "v1",
            "hello world",
        );
        let b = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_A,
            "gpt-4-embed",
            "v2",
            "hello world",
        );
        assert_ne!(a, b);
    }

    #[test]
    fn field_boundary_is_not_ambiguous() {
        let a = CacheKey::derive(OWNER_A, "openai", SCOPE_A, "gpt", "4-embed", "x");
        let b = CacheKey::derive(OWNER_A, "openai", SCOPE_A, "gpt-4", "embed", "x");
        assert_ne!(a, b);
    }

    #[test]
    fn different_owner_produces_different_key() {
        let a = CacheKey::derive(OWNER_A, "openai", SCOPE_A, "gpt-4-embed", "v1", "same text");
        let b = CacheKey::derive(OWNER_B, "openai", SCOPE_A, "gpt-4-embed", "v1", "same text");
        assert_ne!(a, b, "two different callers must never share a cache entry");
    }

    #[test]
    fn different_provider_produces_different_key() {
        let a = CacheKey::derive(OWNER_A, "openai", SCOPE_A, "embed-v1", "v1", "same text");
        let b = CacheKey::derive(OWNER_A, "mistral", SCOPE_A, "embed-v1", "v1", "same text");
        assert_ne!(
            a, b,
            "two providers with an identically-named model must never collide"
        );
    }

    #[test]
    fn different_cache_scope_produces_different_key() {
        // The whole point of the field: an operator repointing an adapter at
        // a self-hosted endpoint, or a caller naming a different region /
        // project / deployment, must get a cold miss rather than a vector
        // computed by some other set of weights.
        let a = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_A,
            "text-embedding-3-small",
            "v1",
            "same text",
        );
        let b = CacheKey::derive(
            OWNER_A,
            "openai",
            SCOPE_B,
            "text-embedding-3-small",
            "v1",
            "same text",
        );
        assert_ne!(
            a, b,
            "two different upstream endpoints must never share a cache entry"
        );
    }

    #[test]
    fn cache_scope_field_boundary_is_not_ambiguous() {
        let a = CacheKey::derive(OWNER_A, "openai", "a", "bc", "v1", "x");
        let b = CacheKey::derive(OWNER_A, "openai", "ab", "c", "v1", "x");
        assert_ne!(
            a, b,
            "the scope/model boundary must be unambiguous, like every other field pair"
        );
    }

    #[test]
    fn derive_image_same_inputs_produce_same_key() {
        let a = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        let b = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn derive_image_key_never_collides_with_a_text_key_of_identical_bytes() {
        // Guards against the base64 payload of an image happening to match
        // the literal text of an unrelated cache entry -- the "image\0"
        // domain-separation byte inside derive_image must make this
        // impossible even if every other field lines up exactly.
        let text_key = CacheKey::derive(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "YmFzZTY0",
        );
        let image_key = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        assert_ne!(text_key, image_key);
    }

    #[test]
    fn derive_image_different_mime_type_produces_different_key() {
        let a = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        let b = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/jpeg",
            "YmFzZTY0",
        );
        assert_ne!(
            a, b,
            "the same bytes decoded under a different mime type are a different image"
        );
    }

    #[test]
    fn derive_image_different_cache_scope_produces_different_key() {
        let a = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_A,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        let b = CacheKey::derive_image(
            OWNER_A,
            "gemini",
            SCOPE_B,
            "gemini-embedding-2",
            "v1",
            "image/png",
            "YmFzZTY0",
        );
        assert_ne!(
            a, b,
            "the image path needs the same endpoint isolation the text path gets"
        );
    }
}

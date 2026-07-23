/// Hashes a caller's raw provider API key into an opaque tenant identifier.
/// The raw key is never stored — only this hash ever appears in a `CacheKey`.
pub fn derive_owner_id(api_key: &str) -> [u8; 32] {
    *blake3::hash(api_key.as_bytes()).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_produces_same_owner_id() {
        assert_eq!(derive_owner_id("sk-abc123"), derive_owner_id("sk-abc123"));
    }

    #[test]
    fn different_keys_produce_different_owner_ids() {
        assert_ne!(derive_owner_id("sk-abc123"), derive_owner_id("sk-xyz789"));
    }
}

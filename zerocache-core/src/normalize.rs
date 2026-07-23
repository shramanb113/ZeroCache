/// Trims leading/trailing whitespace and collapses internal whitespace runs
/// (spaces, tabs, newlines) to a single space. Applied identically before
/// both hashing (the cache key) and sending to the real provider, so a
/// cached vector always corresponds to exactly what was embedded — this is
/// not just a hit-rate optimization, it's a correctness requirement: caching
/// under a normalized key while sending the raw text to the provider would
/// serve a vector that doesn't match what the caller's raw text would have
/// produced.
///
/// Deliberately whitespace-only: no case-folding, no Unicode normalization.
/// Those change semantics in ways this function does not decide.
pub fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_leading_and_trailing_whitespace() {
        assert_eq!(normalize_text("  hello world  "), "hello world");
    }

    #[test]
    fn collapses_internal_whitespace_runs() {
        assert_eq!(normalize_text("hello    world"), "hello world");
    }

    #[test]
    fn collapses_newlines_and_tabs_too() {
        assert_eq!(normalize_text("hello\n\tworld"), "hello world");
    }

    #[test]
    fn already_normalized_text_is_unchanged() {
        assert_eq!(normalize_text("hello world"), "hello world");
    }

    #[test]
    fn empty_string_stays_empty() {
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn does_not_change_casing() {
        assert_eq!(normalize_text("Hello World"), "Hello World");
    }
}

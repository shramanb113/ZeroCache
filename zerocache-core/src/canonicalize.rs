/// Canonicalizes text for cache-key derivation only -- NOT for what is sent
/// to the provider. Two inputs that differ only in casing, Unicode
/// composition form, quote/dash style, or trailing sentence punctuation
/// canonicalize to the same string and therefore share a cache entry, while
/// the vector stored under that entry is still a real embedding of some
/// caller's actual (whitespace-`normalize_text`ed but otherwise untouched)
/// text -- see the split in `zerocache-http`'s `embed_batch`.
///
/// Distinct from `normalize_text`, which is deliberately whitespace-only
/// because it feeds the provider and must not change embedding semantics.
pub fn canonicalize_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    let folded: String = text
        .nfc()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'', // ‘ ’  curly single quotes
            '\u{201C}' | '\u{201D}' => '"',  // “ ”  curly double quotes
            '\u{2013}' | '\u{2014}' => '-',  // – —  en / em dash
            other => other,
        })
        .collect();

    let collapsed = folded
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    collapsed
        .trim_end_matches(|c: char| {
            matches!(c, '.' | ',' | ';' | ':' | '!' | '?') || c.is_whitespace()
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_ascii() {
        assert_eq!(canonicalize_text("Hello World"), "hello world");
    }

    #[test]
    fn collapses_and_trims_whitespace() {
        assert_eq!(canonicalize_text("  hello\n\tworld  "), "hello world");
    }

    #[test]
    fn nfc_normalizes_decomposed_unicode() {
        // "e" + U+0301 COMBINING ACUTE ACCENT vs the precomposed "é".
        assert_eq!(canonicalize_text("cafe\u{0301}"), canonicalize_text("café"));
        assert_eq!(canonicalize_text("cafe\u{0301}"), "café");
    }

    #[test]
    fn folds_curly_quotes_to_ascii() {
        assert_eq!(
            canonicalize_text("\u{2018}it\u{2019}s\u{201C}quoted\u{201D}"),
            "'it's\"quoted\""
        );
    }

    #[test]
    fn folds_en_and_em_dashes_to_hyphen() {
        assert_eq!(canonicalize_text("a\u{2013}b\u{2014}c"), "a-b-c");
    }

    #[test]
    fn trims_trailing_sentence_punctuation() {
        assert_eq!(canonicalize_text("hello world."), "hello world");
        assert_eq!(canonicalize_text("really?!"), "really");
        assert_eq!(canonicalize_text("a, b, c,"), "a, b, c");
    }

    #[test]
    fn trims_trailing_punctuation_with_interleaved_space() {
        assert_eq!(canonicalize_text("what is this ? "), "what is this");
    }

    #[test]
    fn does_not_strip_internal_punctuation() {
        assert_eq!(canonicalize_text("a.b.c"), "a.b.c");
        assert_eq!(canonicalize_text("foo, bar and baz"), "foo, bar and baz");
    }

    #[test]
    fn empty_and_punctuation_only_strings() {
        assert_eq!(canonicalize_text(""), "");
        assert_eq!(canonicalize_text("   "), "");
        assert_eq!(canonicalize_text("?!."), "");
    }

    #[test]
    fn combined_transformations() {
        assert_eq!(
            canonicalize_text("  The Cafe\u{0301}\u{2014}really? "),
            "the café-really"
        );
    }

    #[test]
    fn already_canonical_text_is_unchanged() {
        assert_eq!(canonicalize_text("hello world"), "hello world");
    }
}

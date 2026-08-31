//! Anthropic `/v1/messages` request handling for the completion cache: an
//! order-independent canonical serialization of the output-affecting request
//! fields (for the cache key) and a determinism gate.
//!
//! The OpenAI-shaped equivalents live in `completion.rs`; this is a separate
//! module because the wire shape differs (top-level `system`, block `content`,
//! `max_tokens` required, no `seed`, no `n`).

use serde_json::Value;

use crate::canonical::canonical_value;

/// Top-level Messages fields that do not change the response and so must not
/// be in the cache key. `model` is excluded here only because it is hashed as
/// its own dedicated key field. Denylist, not allowlist — an unrecognized
/// field stays in the key (over-key = a wasted miss; under-key = a wrong
/// answer).
const KEY_IRRELEVANT_FIELDS: &[&str] = &["model", "stream", "metadata"];

/// Serializes the output-affecting parts of an Anthropic `/v1/messages`
/// request into a stable string for cache-key derivation. Two requests that
/// differ only in object-key order, integer-vs-float spelling of the same
/// number, or the `KEY_IRRELEVANT_FIELDS` above produce byte-identical
/// output; a difference in `messages`, `system`, `tools`, or any generation
/// parameter does not.
pub fn canonicalize_messages_request(request: &Value) -> String {
    let mut root = request.clone();
    if let Value::Object(map) = &mut root {
        for field in KEY_IRRELEVANT_FIELDS {
            map.remove(*field);
        }
    }
    serde_json::to_string(&canonical_value(&root)).unwrap_or_default()
}

/// Whether a `/v1/messages` request is deterministic enough that a cached
/// response for the same key is a faithful substitute for calling Anthropic
/// again.
///
/// Requires: non-empty `messages`; an explicit `temperature` whose `as_f64()`
/// is `Some(0.0)` (so an integer `0` and `0.0` both qualify — the same test
/// `completion_request_is_cacheable` uses); and `thinking` absent, `null`, or
/// `{"type":"disabled"}` (adaptive/enabled extended thinking is sampled even
/// at temperature 0). Anthropic has no `seed` and no `n`, so neither is
/// checked. `stream` is ignored — a delivery concern, not a determinism one.
pub fn messages_request_is_cacheable(request: &Value) -> bool {
    let has_nonempty_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|m| !m.is_empty());
    if !has_nonempty_messages {
        return false;
    }

    if request.get("temperature").and_then(Value::as_f64) != Some(0.0) {
        return false;
    }

    match request.get("thinking") {
        None | Some(Value::Null) => true,
        Some(t) => t.get("type").and_then(Value::as_str) == Some("disabled"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> Value {
        json!({
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "temperature": 0
        })
    }

    // ---- canonicalize_messages_request ----

    #[test]
    fn top_level_key_order_does_not_matter() {
        let a = json!({"messages": [], "system": "s", "max_tokens": 10});
        let b = json!({"max_tokens": 10, "system": "s", "messages": []});
        assert_eq!(
            canonicalize_messages_request(&a),
            canonicalize_messages_request(&b)
        );
    }

    #[test]
    fn nested_key_order_does_not_matter() {
        let a = json!({"messages": [], "tool_choice": {"type": "tool", "name": "x"}});
        let b = json!({"messages": [], "tool_choice": {"name": "x", "type": "tool"}});
        assert_eq!(
            canonicalize_messages_request(&a),
            canonicalize_messages_request(&b)
        );
    }

    #[test]
    fn model_stream_and_metadata_are_removed_from_the_canonical_form() {
        let bare = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
        let noisy = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "model": "claude-opus-4-6",
            "stream": true,
            "metadata": {"user_id": "abc"}
        });
        assert_eq!(
            canonicalize_messages_request(&bare),
            canonicalize_messages_request(&noisy)
        );
    }

    #[test]
    fn system_tools_and_max_tokens_are_retained() {
        let a = json!({"messages": [], "system": "be terse", "tools": [{"name": "t"}], "max_tokens": 8});
        let b = json!({"messages": [], "max_tokens": 8});
        assert_ne!(
            canonicalize_messages_request(&a),
            canonicalize_messages_request(&b)
        );
    }

    #[test]
    fn integer_and_float_temperature_zero_canonicalize_equal() {
        let a = json!({"messages": [], "temperature": 0});
        let b = json!({"messages": [], "temperature": 0.0});
        assert_eq!(
            canonicalize_messages_request(&a),
            canonicalize_messages_request(&b)
        );
    }

    #[test]
    fn a_stop_sequences_reorder_changes_the_canonical_form() {
        // Arrays are kept in order — [a,b] and [b,a] are different requests.
        let a = json!({"messages": [], "stop_sequences": ["END", "STOP"]});
        let b = json!({"messages": [], "stop_sequences": ["STOP", "END"]});
        assert_ne!(
            canonicalize_messages_request(&a),
            canonicalize_messages_request(&b)
        );
    }

    // ---- messages_request_is_cacheable ----

    #[test]
    fn temperature_zero_with_messages_is_cacheable() {
        assert!(messages_request_is_cacheable(&base()));
    }

    #[test]
    fn integer_zero_and_float_zero_temperature_both_qualify() {
        let mut r = base();
        r["temperature"] = json!(0.0);
        assert!(messages_request_is_cacheable(&r));
    }

    #[test]
    fn temperature_one_is_not_cacheable() {
        let mut r = base();
        r["temperature"] = json!(1);
        assert!(!messages_request_is_cacheable(&r));
    }

    #[test]
    fn temperature_absent_is_not_cacheable() {
        let mut r = base();
        r.as_object_mut().unwrap().remove("temperature");
        assert!(!messages_request_is_cacheable(&r));
    }

    #[test]
    fn empty_messages_is_not_cacheable() {
        let mut r = base();
        r["messages"] = json!([]);
        assert!(!messages_request_is_cacheable(&r));
    }

    #[test]
    fn enabled_or_adaptive_thinking_is_not_cacheable_even_at_temperature_zero() {
        let mut r = base();
        r["thinking"] = json!({"type": "enabled", "budget_tokens": 1024});
        assert!(!messages_request_is_cacheable(&r));
    }

    #[test]
    fn disabled_thinking_is_cacheable() {
        let mut r = base();
        r["thinking"] = json!({"type": "disabled"});
        assert!(messages_request_is_cacheable(&r));
    }

    #[test]
    fn null_thinking_is_treated_as_absent() {
        let mut r = base();
        r["thinking"] = json!(null);
        assert!(messages_request_is_cacheable(&r));
    }

    #[test]
    fn the_stream_flag_alone_does_not_flip_a_cacheable_request() {
        let mut r = base();
        r["stream"] = json!(true);
        assert!(messages_request_is_cacheable(&r));
    }
}

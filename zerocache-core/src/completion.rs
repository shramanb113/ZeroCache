//! Chat-completion request handling for the completion cache: a canonical,
//! order-independent serialization of the output-affecting parts of a
//! request (for the cache key) and a determinism gate that decides whether a
//! request may be cached at all.
//!
//! Both operate on the already-parsed OpenAI `/v1/chat/completions` JSON
//! body -- turning an HTTP body into a `serde_json::Value` and pulling out
//! `model` is wire-shape translation and lives in `zerocache-http`; deciding
//! *which fields change the completion* is domain knowledge and lives here,
//! next to `canonicalize_text`.

use serde_json::{Map, Number, Value};

/// Top-level request fields that do NOT change the completion the provider
/// returns, and so must not be part of the cache key. `model` is excluded
/// here only because it is hashed as its own dedicated key field.
///
/// This is a denylist, not an allowlist, on purpose: an unrecognized field
/// stays in the key. Over-keying on a harmless field costs an unnecessary
/// miss; under-keying on one that does affect output serves a wrong
/// completion. The whole codebase leans this way (see `EmbeddingProvider::
/// cache_scope`'s "over-invalidate rather than risk under-invalidating").
const KEY_IRRELEVANT_FIELDS: &[&str] = &[
    "model",
    "user",
    "stream",
    "stream_options",
    "store",
    "metadata",
];

/// Serializes the output-affecting parts of a chat-completion request into a
/// stable string for cache-key derivation. Two requests that differ only in
/// object-key order, integer-vs-float spelling of the same number, or the
/// `KEY_IRRELEVANT_FIELDS` above produce byte-identical output; any
/// difference in messages, tools, or a generation parameter does not.
pub fn canonicalize_completion_request(request: &Value) -> String {
    let mut root = request.clone();
    if let Value::Object(map) = &mut root {
        for field in KEY_IRRELEVANT_FIELDS {
            map.remove(*field);
        }
    }
    serde_json::to_string(&canonical_value(&root)).unwrap_or_default()
}

/// Whether a request is deterministic enough that a cached completion for the
/// same key is a faithful substitute for calling the provider again.
///
/// Deliberately does not look at `stream`: streaming is a response-delivery
/// concern resolved at the transport layer, not a property of whether the
/// completion itself is reproducible.
pub fn completion_request_is_cacheable(request: &Value) -> bool {
    let has_nonempty_messages = request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|m| !m.is_empty());
    if !has_nonempty_messages {
        return false;
    }

    // n: absent or null means the provider default of 1; an explicit value
    // other than 1 asks for multiple independent samples, which no single
    // cached completion can stand in for.
    match request.get("n") {
        None | Some(Value::Null) => {}
        Some(v) if v.as_u64() == Some(1) => {}
        Some(_) => return false,
    }

    let temperature_is_zero = request.get("temperature").and_then(Value::as_f64) == Some(0.0);
    let seed_is_set = request
        .get("seed")
        .is_some_and(|v| v.is_i64() || v.is_u64());

    temperature_is_zero || seed_is_set
}

/// Recursively rewrites a JSON value into canonical form: object keys sorted,
/// numbers normalized so `0` and `0.0` compare equal, arrays left in order.
fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for key in keys {
                out.insert(key.clone(), canonical_value(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        Value::Number(n) => canonical_number(n),
        other => other.clone(),
    }
}

/// Collapses integer-valued numbers to a single spelling: `0`, `0.0`, and
/// `-0.0` all become `0`; genuine fractions are kept as-is.
fn canonical_number(n: &Number) -> Value {
    if n.is_i64() || n.is_u64() {
        return Value::Number(n.clone());
    }
    if let Some(f) = n.as_f64() {
        if f.is_finite() && f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0 {
            return Value::Number(Number::from(f as i64));
        }
        if let Some(norm) = Number::from_f64(f) {
            return Value::Number(norm);
        }
    }
    Value::Number(n.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- canonicalize_completion_request ----

    #[test]
    fn identical_requests_canonicalize_equal() {
        let a = json!({"messages":[{"role":"user","content":"hi"}],"temperature":0});
        let b = json!({"messages":[{"role":"user","content":"hi"}],"temperature":0});
        assert_eq!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn top_level_key_order_does_not_matter() {
        let a = json!({"messages":[{"role":"user","content":"hi"}],"temperature":0});
        let b = json!({"temperature":0,"messages":[{"role":"user","content":"hi"}]});
        assert_eq!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn nested_object_key_order_does_not_matter() {
        let a = json!({
            "messages": [],
            "response_format": {"type":"json_schema","json_schema":{"name":"x","strict":true}}
        });
        let b = json!({
            "messages": [],
            "response_format": {"json_schema":{"strict":true,"name":"x"},"type":"json_schema"}
        });
        assert_eq!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn key_irrelevant_fields_do_not_affect_canonical_form() {
        let base = json!({"messages":[{"role":"user","content":"hi"}],"temperature":0});
        let noisy = json!({
            "messages":[{"role":"user","content":"hi"}],
            "temperature":0,
            "model":"gpt-4o",
            "user":"alice",
            "stream":true,
            "stream_options":{"include_usage":true},
            "store":true,
            "metadata":{"trace":"abc"}
        });
        assert_eq!(
            canonicalize_completion_request(&base),
            canonicalize_completion_request(&noisy)
        );
    }

    #[test]
    fn integer_and_float_spelling_of_the_same_number_canonicalize_equal() {
        let a = json!({"messages":[],"temperature":0});
        let b = json!({"messages":[],"temperature":0.0});
        assert_eq!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn message_content_difference_changes_canonical_form() {
        let a = json!({"messages":[{"role":"user","content":"hello"}]});
        let b = json!({"messages":[{"role":"user","content":"HELLO"}]});
        assert_ne!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn message_order_is_significant() {
        let a = json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}]});
        let b = json!({"messages":[{"role":"user","content":"u"},{"role":"system","content":"s"}]});
        assert_ne!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn tool_definitions_are_part_of_the_canonical_form() {
        let with_tools = json!({
            "messages": [],
            "tools": [{"type":"function","function":{"name":"get_weather"}}]
        });
        let without = json!({"messages": []});
        assert_ne!(
            canonicalize_completion_request(&with_tools),
            canonicalize_completion_request(&without)
        );
    }

    #[test]
    fn an_output_affecting_param_changes_the_canonical_form() {
        let a = json!({"messages":[],"max_tokens":100});
        let b = json!({"messages":[],"max_tokens":200});
        assert_ne!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    #[test]
    fn an_unrecognized_param_is_kept_in_the_canonical_form() {
        let a = json!({"messages":[],"some_future_param":true});
        let b = json!({"messages":[]});
        assert_ne!(
            canonicalize_completion_request(&a),
            canonicalize_completion_request(&b)
        );
    }

    // ---- completion_request_is_cacheable ----

    #[test]
    fn temperature_zero_is_cacheable() {
        assert!(completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"temperature":0
        })));
    }

    #[test]
    fn seed_present_is_cacheable_even_without_temperature_zero() {
        assert!(completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"seed":42
        })));
    }

    #[test]
    fn nonzero_temperature_without_a_seed_is_not_cacheable() {
        assert!(!completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"temperature":0.7
        })));
    }

    #[test]
    fn neither_temperature_nor_seed_is_not_cacheable() {
        assert!(!completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}]
        })));
    }

    #[test]
    fn n_greater_than_one_is_not_cacheable_even_at_temperature_zero() {
        assert!(!completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"temperature":0,"n":2
        })));
    }

    #[test]
    fn n_explicitly_one_at_temperature_zero_is_cacheable() {
        assert!(completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"temperature":0,"n":1
        })));
    }

    #[test]
    fn a_request_without_messages_is_not_cacheable() {
        assert!(!completion_request_is_cacheable(&json!({"temperature":0})));
    }

    #[test]
    fn a_request_with_empty_messages_is_not_cacheable() {
        assert!(!completion_request_is_cacheable(
            &json!({"messages":[],"temperature":0})
        ));
    }

    #[test]
    fn the_stream_flag_does_not_affect_the_cacheability_gate() {
        assert!(completion_request_is_cacheable(&json!({
            "messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true
        })));
    }
}

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

use serde_json::Value;

use crate::canonical::canonical_value;

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

/// Which span of a chat request the semantic tier embeds as "fuzzy"; everything
/// outside it is still matched exactly. `LastUser` is the tightest, and default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatchUnit {
    LastUser = 0,
    SystemAndLastUser = 1,
    FullConversation = 2,
}

/// A message's `content` as text: a bare string, or the `text` parts of an
/// array joined with '\n'. Non-text parts and other shapes yield `None`.
fn message_text(message: &Value) -> Option<String> {
    match message.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let joined = parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

fn messages(request: &Value) -> &[Value] {
    request
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn last_user_text(request: &Value) -> Option<String> {
    messages(request)
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(message_text)
}

/// The text to embed for `unit`, or `None` when the span is absent/blank (the
/// caller then skips the semantic tier for this request).
pub fn completion_fuzzy_text(request: &Value, unit: MatchUnit) -> Option<String> {
    let text = match unit {
        MatchUnit::LastUser => last_user_text(request)?,
        MatchUnit::SystemAndLastUser => {
            let user = last_user_text(request)?;
            let mut parts: Vec<String> = messages(request)
                .iter()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
                .filter_map(message_text)
                .collect();
            parts.push(user);
            parts.join("\n")
        }
        MatchUnit::FullConversation => {
            let lines: Vec<String> = messages(request)
                .iter()
                .filter_map(|m| {
                    let role = m.get("role").and_then(Value::as_str)?;
                    let text = message_text(m)?;
                    Some(format!("{role}: {text}"))
                })
                .collect();
            if lines.is_empty() {
                return None;
            }
            lines.join("\n")
        }
    };
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// The canonical form with the fuzzy span (per `unit`) blanked. Feeds
/// `coarse_key_hash` — the gate a semantic candidate must still match exactly.
pub fn canonicalize_completion_request_coarse(request: &Value, unit: MatchUnit) -> String {
    let mut root = request.clone();
    if let Value::Object(map) = &mut root {
        for field in KEY_IRRELEVANT_FIELDS {
            map.remove(*field);
        }
    }

    match unit {
        MatchUnit::LastUser | MatchUnit::SystemAndLastUser => {
            if let Some(Value::Array(msgs)) = root.get_mut("messages") {
                if let Some(m) = msgs
                    .iter_mut()
                    .rev()
                    .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                {
                    blank_content(m);
                }
                if unit == MatchUnit::SystemAndLastUser {
                    for m in msgs
                        .iter_mut()
                        .filter(|m| m.get("role").and_then(Value::as_str) == Some("system"))
                    {
                        blank_content(m);
                    }
                }
            }
        }
        MatchUnit::FullConversation => {
            if let Value::Object(map) = &mut root {
                map.remove("messages");
            }
        }
    }

    serde_json::to_string(&canonical_value(&root)).unwrap_or_default()
}

fn blank_content(msg: &mut Value) {
    if let Value::Object(obj) = msg {
        obj.insert("content".to_string(), Value::String(String::new()));
    }
}

/// blake3 of the coarse canonical form, `unit` discriminant folded in so
/// changing `ZEROCACHE_SEMANTIC_MATCH_UNIT` can't produce a cross-unit match.
pub fn coarse_key_hash(request: &Value, unit: MatchUnit) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[unit as u8]);
    hasher.update(b"\0");
    hasher.update(canonicalize_completion_request_coarse(request, unit).as_bytes());
    *hasher.finalize().as_bytes()
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

    // ---- completion_fuzzy_text ----

    #[test]
    fn fuzzy_last_user_returns_the_final_user_message_string() {
        let req = json!({"messages":[
            {"role":"system","content":"be terse"},
            {"role":"user","content":"first"},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":"how do I reset my password?"}
        ]});
        assert_eq!(
            completion_fuzzy_text(&req, MatchUnit::LastUser).as_deref(),
            Some("how do I reset my password?")
        );
    }

    #[test]
    fn fuzzy_last_user_flattens_array_of_text_parts() {
        let req = json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":"line one"},
                {"type":"text","text":"line two"},
                {"type":"image_url","image_url":{"url":"data:x"}}
            ]}
        ]});
        assert_eq!(
            completion_fuzzy_text(&req, MatchUnit::LastUser).as_deref(),
            Some("line one\nline two")
        );
    }

    #[test]
    fn fuzzy_returns_none_when_there_is_no_user_message() {
        let req = json!({"messages":[{"role":"system","content":"hi"}]});
        assert_eq!(completion_fuzzy_text(&req, MatchUnit::LastUser), None);
        assert_eq!(
            completion_fuzzy_text(&req, MatchUnit::SystemAndLastUser),
            None
        );
    }

    #[test]
    fn fuzzy_returns_none_for_a_whitespace_only_user_message() {
        let req = json!({"messages":[{"role":"user","content":"   \n"}]});
        assert_eq!(completion_fuzzy_text(&req, MatchUnit::LastUser), None);
    }

    #[test]
    fn fuzzy_system_and_last_user_joins_system_then_user() {
        let req = json!({"messages":[
            {"role":"system","content":"persona A"},
            {"role":"system","content":"persona B"},
            {"role":"user","content":"the question"}
        ]});
        assert_eq!(
            completion_fuzzy_text(&req, MatchUnit::SystemAndLastUser).as_deref(),
            Some("persona A\npersona B\nthe question")
        );
    }

    #[test]
    fn fuzzy_full_conversation_joins_every_message_with_role_labels() {
        let req = json!({"messages":[
            {"role":"system","content":"s"},
            {"role":"user","content":"u1"},
            {"role":"assistant","content":"a1"},
            {"role":"user","content":"u2"}
        ]});
        assert_eq!(
            completion_fuzzy_text(&req, MatchUnit::FullConversation).as_deref(),
            Some("system: s\nuser: u1\nassistant: a1\nuser: u2")
        );
    }

    // ---- canonicalize_completion_request_coarse / coarse_key_hash ----

    fn base_chat() -> serde_json::Value {
        json!({
            "model": "gpt-4o",
            "messages": [
                {"role":"system","content":"you are a support bot"},
                {"role":"user","content":"how do I reset my password?"}
            ],
            "temperature": 0,
            "max_tokens": 256
        })
    }

    #[test]
    fn coarse_hash_ignores_a_change_to_only_the_last_user_message() {
        let a = base_chat();
        let mut b = base_chat();
        b["messages"][1]["content"] = json!("how can I reset my password??");
        assert_eq!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&b, MatchUnit::LastUser),
        );
    }

    #[test]
    fn coarse_hash_changes_when_the_system_prompt_changes_under_last_user() {
        let a = base_chat();
        let mut b = base_chat();
        b["messages"][0]["content"] = json!("you are a TERSE support bot");
        assert_ne!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&b, MatchUnit::LastUser),
        );
    }

    #[test]
    fn coarse_hash_changes_when_a_generation_param_changes() {
        let a = base_chat();
        let mut b = base_chat();
        b["max_tokens"] = json!(512);
        assert_ne!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&b, MatchUnit::LastUser),
        );
    }

    #[test]
    fn coarse_hash_ignores_key_irrelevant_fields() {
        let a = base_chat();
        let mut b = base_chat();
        b["user"] = json!("alice");
        b["stream"] = json!(true);
        assert_eq!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&b, MatchUnit::LastUser),
        );
    }

    #[test]
    fn system_and_last_user_also_blanks_the_system_prompt() {
        let a = base_chat();
        let mut b = base_chat();
        b["messages"][0]["content"] = json!("a completely different persona");
        assert_eq!(
            coarse_key_hash(&a, MatchUnit::SystemAndLastUser),
            coarse_key_hash(&b, MatchUnit::SystemAndLastUser),
        );
        assert_ne!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&b, MatchUnit::LastUser),
        );
    }

    #[test]
    fn full_conversation_drops_messages_but_keeps_params_and_tools() {
        let a = base_chat();
        let mut b = base_chat();
        b["messages"] = json!([{"role":"user","content":"totally different history"}]);
        assert_eq!(
            coarse_key_hash(&a, MatchUnit::FullConversation),
            coarse_key_hash(&b, MatchUnit::FullConversation),
        );
        let mut c = base_chat();
        c["tools"] = json!([{"type":"function","function":{"name":"x"}}]);
        assert_ne!(
            coarse_key_hash(&a, MatchUnit::FullConversation),
            coarse_key_hash(&c, MatchUnit::FullConversation),
        );
    }

    #[test]
    fn the_match_unit_discriminant_is_part_of_the_coarse_hash() {
        let a = base_chat();
        assert_ne!(
            coarse_key_hash(&a, MatchUnit::LastUser),
            coarse_key_hash(&a, MatchUnit::FullConversation),
        );
    }
}

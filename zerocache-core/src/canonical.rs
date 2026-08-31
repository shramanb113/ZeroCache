//! Order-independent JSON canonicalization shared by the completion-cache
//! canonicalizers (`completion.rs`, `messages.rs`): object keys sorted,
//! integer-valued numbers collapsed to one spelling, arrays left in order.

use serde_json::{Map, Number, Value};

/// Recursively rewrites a JSON value into canonical form: object keys sorted,
/// numbers normalized so `0` and `0.0` compare equal, arrays left in order.
pub(crate) fn canonical_value(value: &Value) -> Value {
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
pub(crate) fn canonical_number(n: &Number) -> Value {
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

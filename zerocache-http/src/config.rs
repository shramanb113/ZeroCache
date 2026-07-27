use std::time::Duration;

pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";
pub const DEFAULT_MISTRAL_BASE_URL: &str = "https://api.mistral.ai";
pub const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://router.huggingface.co/hf-inference";

pub enum StorageBackend {
    // Embedded, single-process. Fine for local dev or a single-replica
    // deployment; cannot be shared across multiple Kubernetes pods.
    Sled,
    // Shared, network-accessible. Required for multi-replica deployments so
    // every pod hits the same cache instead of each keeping a private one.
    Redis,
}

pub struct Config {
    pub port: u16,
    pub storage_backend: StorageBackend,
    pub storage_path: String,
    pub redis_url: String,
    pub ttl: Option<Duration>,
    pub openai_base_url: String,
    pub mistral_base_url: String,
    pub gemini_base_url: String,
    pub huggingface_base_url: String,
}

/// Resolves an optional env-var override to a base URL, falling back to
/// `default` when the var is unset or empty. Pulled out as a pure function
/// (rather than inlined per call site) so it's unit-testable without
/// mutating real process env vars -- same reasoning as parse_ttl_seconds
/// below. An empty string is treated the same as unset, matching
/// parse_ttl_seconds's treatment of an empty ZEROCACHE_TTL_SECONDS, rather
/// than producing an empty base URL that would fail confusingly deep
/// inside reqwest.
fn base_url_or_default(raw: Option<&str>, default: &str) -> String {
    match raw {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => default.to_string(),
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("ZEROCACHE_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            storage_backend: match std::env::var("ZEROCACHE_STORAGE_BACKEND").as_deref() {
                Ok("redis") => StorageBackend::Redis,
                _ => StorageBackend::Sled,
            },
            storage_path: std::env::var("ZEROCACHE_STORAGE_PATH").unwrap_or_else(|_| "./data".into()),
            redis_url: std::env::var("ZEROCACHE_REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            ttl: parse_ttl_seconds(std::env::var("ZEROCACHE_TTL_SECONDS").ok().as_deref()),
            openai_base_url: base_url_or_default(
                std::env::var("ZEROCACHE_OPENAI_BASE_URL").ok().as_deref(),
                DEFAULT_OPENAI_BASE_URL,
            ),
            mistral_base_url: base_url_or_default(
                std::env::var("ZEROCACHE_MISTRAL_BASE_URL").ok().as_deref(),
                DEFAULT_MISTRAL_BASE_URL,
            ),
            gemini_base_url: base_url_or_default(
                std::env::var("ZEROCACHE_GEMINI_BASE_URL").ok().as_deref(),
                DEFAULT_GEMINI_BASE_URL,
            ),
            huggingface_base_url: base_url_or_default(
                std::env::var("ZEROCACHE_HUGGINGFACE_BASE_URL").ok().as_deref(),
                DEFAULT_HUGGINGFACE_BASE_URL,
            ),
        }
    }
}

/// Parses the raw `ZEROCACHE_TTL_SECONDS` value into an optional TTL.
///
/// `0` is treated as "unset" rather than "expire immediately"/"reject writes",
/// since the two storage backends disagree on what a zero-second TTL means:
/// Redis's `SET...EX 0` is rejected outright (`ERR invalid expire time`),
/// while sled would treat it as instant-expiry, silently producing 0% hit
/// rate. An unparseable value (empty string, non-numeric, negative) is
/// likewise treated as "unset". Both cases print a startup warning so an
/// operator who meant to configure a TTL isn't left silently guessing.
fn parse_ttl_seconds(raw: Option<&str>) -> Option<Duration> {
    let v = raw?;
    match v.parse::<u64>() {
        Ok(secs) if secs > 0 => Some(Duration::from_secs(secs)),
        Ok(_) => {
            eprintln!(
                "warning: ZEROCACHE_TTL_SECONDS=0 is ambiguous (Redis rejects it, sled treats it as instant-expiry) -- ignoring, entries will never expire"
            );
            None
        }
        Err(_) => {
            eprintln!(
                "warning: ZEROCACHE_TTL_SECONDS='{v}' is not a valid positive integer -- ignoring, entries will never expire"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttl_zero_is_treated_as_unset() {
        assert_eq!(parse_ttl_seconds(Some("0")), None);
    }

    #[test]
    fn ttl_empty_string_is_treated_as_unset() {
        assert_eq!(parse_ttl_seconds(Some("")), None);
    }

    #[test]
    fn ttl_non_numeric_is_treated_as_unset() {
        assert_eq!(parse_ttl_seconds(Some("abc")), None);
    }

    #[test]
    fn ttl_negative_is_treated_as_unset() {
        assert_eq!(parse_ttl_seconds(Some("-5")), None);
    }

    #[test]
    fn ttl_valid_positive_value_is_parsed() {
        assert_eq!(parse_ttl_seconds(Some("30")), Some(Duration::from_secs(30)));
    }

    #[test]
    fn ttl_unset_var_is_none() {
        assert_eq!(parse_ttl_seconds(None), None);
    }

    #[test]
    fn openai_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(base_url_or_default(None, DEFAULT_OPENAI_BASE_URL), DEFAULT_OPENAI_BASE_URL);
    }

    #[test]
    fn openai_base_url_can_be_overridden() {
        assert_eq!(
            base_url_or_default(Some("http://localhost:11434"), DEFAULT_OPENAI_BASE_URL),
            "http://localhost:11434"
        );
    }

    #[test]
    fn mistral_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(base_url_or_default(None, DEFAULT_MISTRAL_BASE_URL), DEFAULT_MISTRAL_BASE_URL);
    }

    #[test]
    fn mistral_base_url_can_be_overridden() {
        assert_eq!(
            base_url_or_default(Some("http://localhost:11435"), DEFAULT_MISTRAL_BASE_URL),
            "http://localhost:11435"
        );
    }

    #[test]
    fn gemini_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(
            base_url_or_default(None, DEFAULT_GEMINI_BASE_URL),
            DEFAULT_GEMINI_BASE_URL
        );
    }

    #[test]
    fn gemini_base_url_can_be_overridden() {
        assert_eq!(
            base_url_or_default(Some("http://localhost:11436"), DEFAULT_GEMINI_BASE_URL),
            "http://localhost:11436"
        );
    }

    #[test]
    fn huggingface_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(
            base_url_or_default(None, DEFAULT_HUGGINGFACE_BASE_URL),
            DEFAULT_HUGGINGFACE_BASE_URL
        );
    }

    #[test]
    fn huggingface_base_url_can_be_overridden() {
        assert_eq!(
            base_url_or_default(Some("http://localhost:11437"), DEFAULT_HUGGINGFACE_BASE_URL),
            "http://localhost:11437"
        );
    }

    #[test]
    fn empty_base_url_override_is_treated_as_unset() {
        assert_eq!(base_url_or_default(Some(""), DEFAULT_OPENAI_BASE_URL), DEFAULT_OPENAI_BASE_URL);
    }
}

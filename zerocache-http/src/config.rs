use std::time::Duration;

use zerocache_adapters_azure::{AzureAuthMode, DEFAULT_AZURE_FOUNDRY_API_VERSION};
use zerocache_adapters_bedrock::{DEFAULT_BEDROCK_ENDPOINT_TEMPLATE, DEFAULT_BEDROCK_REGION};
use zerocache_adapters_vertexai::{DEFAULT_VERTEX_ENDPOINT_TEMPLATE, DEFAULT_VERTEX_LOCATION};

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
    /// Setting this is what registers the `azure` provider at all -- an Azure
    /// resource name *is* its hostname, so unlike every other provider there
    /// is no meaningful default to fall back to.
    pub azure_openai_base_url: Option<String>,
    pub azure_foundry_base_url: Option<String>,
    pub azure_foundry_api_version: String,
    pub azure_auth_mode: AzureAuthMode,
    pub bedrock_region: String,
    pub bedrock_endpoint_template: String,
    /// `None` means every `vertexai` request's `model` must carry
    /// `<location>/<project>/` itself.
    pub vertex_project: Option<String>,
    pub vertex_location: String,
    pub vertex_endpoint_template: String,
}

/// Resolves an optional env-var override to a string value, falling back to
/// `default` when the var is unset or empty. Used for base URLs (the
/// original motivating case) as well as non-URL values that share the same
/// unset-or-empty-falls-back-to-default shape: `azure_foundry_api_version`,
/// `bedrock_region`, `vertex_location`. Pulled out as a pure function
/// (rather than inlined per call site) so it's unit-testable without
/// mutating real process env vars -- same reasoning as parse_ttl_seconds
/// below. An empty string is treated the same as unset, matching
/// parse_ttl_seconds's treatment of an empty ZEROCACHE_TTL_SECONDS, rather
/// than producing an empty value that would fail confusingly deep inside
/// reqwest (for the URL call sites) or the adapter it's passed to.
fn env_or_default(raw: Option<&str>, default: &str) -> String {
    match raw {
        Some(v) if !v.is_empty() => v.to_string(),
        _ => default.to_string(),
    }
}

/// Reads an env var that has no default. Empty is treated as unset, matching
/// env_or_default and parse_ttl_seconds, rather than producing an empty
/// string that would fail confusingly further down.
fn optional_env(raw: Option<&str>) -> Option<String> {
    match raw {
        Some(v) if !v.is_empty() => Some(v.to_string()),
        _ => None,
    }
}

/// Parses ZEROCACHE_AZURE_AUTH_MODE. Unset defaults to `bearer` silently
/// (that is the documented, recommended mode); an *unrecognized* value warns
/// before falling back, because an operator who typed `apikey` instead of
/// `api-key` would otherwise silently get a mode they did not ask for and see
/// only a 401 from Azure. Same posture as parse_ttl_seconds.
fn parse_azure_auth_mode(raw: Option<&str>) -> AzureAuthMode {
    match raw {
        None | Some("") | Some("bearer") => AzureAuthMode::Bearer,
        Some("api-key") => AzureAuthMode::ApiKey,
        Some(other) => {
            eprintln!(
                "warning: ZEROCACHE_AZURE_AUTH_MODE='{other}' is not recognized (expected 'bearer' or 'api-key') -- falling back to 'bearer'"
            );
            AzureAuthMode::Bearer
        }
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
            storage_path: std::env::var("ZEROCACHE_STORAGE_PATH")
                .unwrap_or_else(|_| "./data".into()),
            redis_url: std::env::var("ZEROCACHE_REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            ttl: parse_ttl_seconds(std::env::var("ZEROCACHE_TTL_SECONDS").ok().as_deref()),
            openai_base_url: env_or_default(
                std::env::var("ZEROCACHE_OPENAI_BASE_URL").ok().as_deref(),
                DEFAULT_OPENAI_BASE_URL,
            ),
            mistral_base_url: env_or_default(
                std::env::var("ZEROCACHE_MISTRAL_BASE_URL").ok().as_deref(),
                DEFAULT_MISTRAL_BASE_URL,
            ),
            gemini_base_url: env_or_default(
                std::env::var("ZEROCACHE_GEMINI_BASE_URL").ok().as_deref(),
                DEFAULT_GEMINI_BASE_URL,
            ),
            huggingface_base_url: env_or_default(
                std::env::var("ZEROCACHE_HUGGINGFACE_BASE_URL")
                    .ok()
                    .as_deref(),
                DEFAULT_HUGGINGFACE_BASE_URL,
            ),
            azure_openai_base_url: optional_env(
                std::env::var("ZEROCACHE_AZURE_OPENAI_BASE_URL")
                    .ok()
                    .as_deref(),
            ),
            azure_foundry_base_url: optional_env(
                std::env::var("ZEROCACHE_AZURE_FOUNDRY_BASE_URL")
                    .ok()
                    .as_deref(),
            ),
            azure_foundry_api_version: env_or_default(
                std::env::var("ZEROCACHE_AZURE_FOUNDRY_API_VERSION")
                    .ok()
                    .as_deref(),
                DEFAULT_AZURE_FOUNDRY_API_VERSION,
            ),
            azure_auth_mode: parse_azure_auth_mode(
                std::env::var("ZEROCACHE_AZURE_AUTH_MODE").ok().as_deref(),
            ),
            bedrock_region: env_or_default(
                std::env::var("ZEROCACHE_BEDROCK_REGION").ok().as_deref(),
                DEFAULT_BEDROCK_REGION,
            ),
            bedrock_endpoint_template: env_or_default(
                std::env::var("ZEROCACHE_BEDROCK_ENDPOINT_TEMPLATE")
                    .ok()
                    .as_deref(),
                DEFAULT_BEDROCK_ENDPOINT_TEMPLATE,
            ),
            vertex_project: optional_env(std::env::var("ZEROCACHE_VERTEX_PROJECT").ok().as_deref()),
            vertex_location: env_or_default(
                std::env::var("ZEROCACHE_VERTEX_LOCATION").ok().as_deref(),
                DEFAULT_VERTEX_LOCATION,
            ),
            vertex_endpoint_template: env_or_default(
                std::env::var("ZEROCACHE_VERTEX_ENDPOINT_TEMPLATE")
                    .ok()
                    .as_deref(),
                DEFAULT_VERTEX_ENDPOINT_TEMPLATE,
            ),
        }
    }
}

/// Normalizes a chat-provider URL to the prefix the chat adapter appends
/// `/chat/completions` to. Deliberately forgiving: a deployer can supply
/// the bare prefix, the prefix with a trailing slash, or the full
/// completions URL pasted straight from the provider's docs -- all three
/// collapse to the same value. Not validated as a URL here (same posture
/// as the ZEROCACHE_*_BASE_URL values): a bad value fails loudly on the
/// first request and is echoed in the startup log.
fn normalize_chat_url(raw: &str) -> String {
    let mut s: &str = raw.trim().trim_end_matches('/');
    if let Some(stripped) = s.strip_suffix("/chat/completions") {
        s = stripped.trim_end_matches('/');
    }
    s.to_string()
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
        assert_eq!(
            env_or_default(None, DEFAULT_OPENAI_BASE_URL),
            DEFAULT_OPENAI_BASE_URL
        );
    }

    #[test]
    fn openai_base_url_can_be_overridden() {
        assert_eq!(
            env_or_default(Some("http://localhost:11434"), DEFAULT_OPENAI_BASE_URL),
            "http://localhost:11434"
        );
    }

    #[test]
    fn mistral_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(
            env_or_default(None, DEFAULT_MISTRAL_BASE_URL),
            DEFAULT_MISTRAL_BASE_URL
        );
    }

    #[test]
    fn mistral_base_url_can_be_overridden() {
        assert_eq!(
            env_or_default(Some("http://localhost:11435"), DEFAULT_MISTRAL_BASE_URL),
            "http://localhost:11435"
        );
    }

    #[test]
    fn gemini_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(
            env_or_default(None, DEFAULT_GEMINI_BASE_URL),
            DEFAULT_GEMINI_BASE_URL
        );
    }

    #[test]
    fn gemini_base_url_can_be_overridden() {
        assert_eq!(
            env_or_default(Some("http://localhost:11436"), DEFAULT_GEMINI_BASE_URL),
            "http://localhost:11436"
        );
    }

    #[test]
    fn huggingface_base_url_defaults_to_real_endpoint_when_unset() {
        assert_eq!(
            env_or_default(None, DEFAULT_HUGGINGFACE_BASE_URL),
            DEFAULT_HUGGINGFACE_BASE_URL
        );
    }

    #[test]
    fn huggingface_base_url_can_be_overridden() {
        assert_eq!(
            env_or_default(Some("http://localhost:11437"), DEFAULT_HUGGINGFACE_BASE_URL),
            "http://localhost:11437"
        );
    }

    #[test]
    fn empty_base_url_override_is_treated_as_unset() {
        assert_eq!(
            env_or_default(Some(""), DEFAULT_OPENAI_BASE_URL),
            DEFAULT_OPENAI_BASE_URL
        );
    }

    #[test]
    fn optional_env_treats_unset_and_empty_the_same() {
        assert_eq!(optional_env(None), None);
        assert_eq!(optional_env(Some("")), None);
        assert_eq!(optional_env(Some("value")), Some("value".to_string()));
    }

    #[test]
    fn azure_auth_mode_defaults_to_bearer() {
        assert_eq!(parse_azure_auth_mode(None), AzureAuthMode::Bearer);
        assert_eq!(parse_azure_auth_mode(Some("")), AzureAuthMode::Bearer);
        assert_eq!(parse_azure_auth_mode(Some("bearer")), AzureAuthMode::Bearer);
    }

    #[test]
    fn azure_auth_mode_api_key_is_recognized() {
        assert_eq!(
            parse_azure_auth_mode(Some("api-key")),
            AzureAuthMode::ApiKey
        );
    }

    #[test]
    fn azure_auth_mode_falls_back_to_bearer_on_an_unrecognized_value() {
        // A typo must not silently select a mode the operator did not ask for
        // and leave them staring at a 401 from Azure.
        assert_eq!(parse_azure_auth_mode(Some("apikey")), AzureAuthMode::Bearer);
    }

    #[test]
    fn bedrock_region_defaults_and_can_be_overridden() {
        assert_eq!(
            env_or_default(None, DEFAULT_BEDROCK_REGION),
            DEFAULT_BEDROCK_REGION
        );
        assert_eq!(
            env_or_default(Some("eu-west-1"), DEFAULT_BEDROCK_REGION),
            "eu-west-1"
        );
    }

    #[test]
    fn vertex_location_defaults_and_can_be_overridden() {
        assert_eq!(
            env_or_default(None, DEFAULT_VERTEX_LOCATION),
            DEFAULT_VERTEX_LOCATION
        );
        assert_eq!(
            env_or_default(Some("europe-west4"), DEFAULT_VERTEX_LOCATION),
            "europe-west4"
        );
    }

    #[test]
    fn normalize_chat_url_leaves_a_bare_prefix_untouched() {
        assert_eq!(
            normalize_chat_url("https://api.groq.com/openai/v1"),
            "https://api.groq.com/openai/v1"
        );
    }

    #[test]
    fn normalize_chat_url_strips_a_trailing_slash() {
        assert_eq!(
            normalize_chat_url("https://api.groq.com/openai/v1/"),
            "https://api.groq.com/openai/v1"
        );
    }

    #[test]
    fn normalize_chat_url_strips_a_trailing_chat_completions_segment() {
        assert_eq!(
            normalize_chat_url("https://api.groq.com/openai/v1/chat/completions"),
            "https://api.groq.com/openai/v1"
        );
    }

    #[test]
    fn normalize_chat_url_strips_both_a_completions_suffix_and_slashes() {
        assert_eq!(
            normalize_chat_url("https://x.example/v1/chat/completions/"),
            "https://x.example/v1"
        );
    }

    #[test]
    fn normalize_chat_url_trims_surrounding_whitespace() {
        assert_eq!(
            normalize_chat_url("  https://x.example/v1  "),
            "https://x.example/v1"
        );
    }

    #[test]
    fn normalize_chat_url_is_idempotent() {
        let once = normalize_chat_url("https://x.example/v1/chat/completions/");
        assert_eq!(normalize_chat_url(&once), once);
    }
}

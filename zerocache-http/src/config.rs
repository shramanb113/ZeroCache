use std::time::Duration;

use zerocache_adapters_azure::{AzureAuthMode, DEFAULT_AZURE_FOUNDRY_API_VERSION};
use zerocache_adapters_bedrock::{DEFAULT_BEDROCK_ENDPOINT_TEMPLATE, DEFAULT_BEDROCK_REGION};
use zerocache_adapters_vertexai::{DEFAULT_VERTEX_ENDPOINT_TEMPLATE, DEFAULT_VERTEX_LOCATION};
use zerocache_core::MatchUnit;

pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";
pub const DEFAULT_SEMANTIC_THRESHOLD: f32 = 0.97;
const SEMANTIC_THRESHOLD_FLOOR: f32 = 0.5;
pub use zerocache_adapters_redis::DEFAULT_SEMANTIC_INDEX_MAXLEN;
pub const DEFAULT_SEMANTIC_POLL_MS: u64 = 2000;
const SEMANTIC_POLL_MS_MIN: u64 = 250;
const SEMANTIC_POLL_MS_MAX: u64 = 60_000;
pub const DEFAULT_MISTRAL_BASE_URL: &str = "https://api.mistral.ai";
pub const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://router.huggingface.co/hf-inference";

/// Built-in OpenAI-wire chat providers, registered even when
/// ZEROCACHE_CHAT_PROVIDERS is unset. Each value is the URL prefix the chat
/// adapter appends `/chat/completions` to -- NOT a bare origin. Unlike the
/// ZEROCACHE_*_BASE_URL embedding vars, this cannot be "origin + /v1":
/// Gemini's OpenAI-compat surface lives under /v1beta/openai. URLs verified
/// against each provider's current docs 2026-08-29; a stale one produces a
/// clean 404 on first use (visible in the startup log), never a wrong
/// answer.
const BUILTIN_CHAT_PROVIDERS: &[(&str, &str)] = &[
    ("openai", "https://api.openai.com/v1"),
    ("mistral", "https://api.mistral.ai/v1"),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    ("groq", "https://api.groq.com/openai/v1"),
    ("deepseek", "https://api.deepseek.com/v1"),
    ("together", "https://api.together.ai/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("xai", "https://api.x.ai/v1"),
    ("fireworks", "https://api.fireworks.ai/inference/v1"),
];

/// Built-in Anthropic `/v1/messages` providers, registered even when
/// ZEROCACHE_MESSAGES_PROVIDERS is unset. Unlike BUILTIN_CHAT_PROVIDERS, each
/// value is a **bare origin** — the adapter appends `/v1/messages` itself —
/// matching the ZEROCACHE_*_BASE_URL embedding-var convention, not the chat
/// "full prefix" one.
const BUILTIN_MESSAGES_PROVIDERS: &[(&str, &str)] = &[("anthropic", "https://api.anthropic.com")];

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
    /// `(name, url_prefix)` pairs: the built-in OpenAI-wire chat providers
    /// merged with any ZEROCACHE_CHAT_PROVIDERS overrides/additions. Order
    /// is unimportant -- main.rs only iterates it.
    pub chat_providers: Vec<(String, String)>,
    /// `(name, bare_origin_url)` pairs: the built-in `anthropic` merged with
    /// any ZEROCACHE_MESSAGES_PROVIDERS overrides/additions. main.rs iterates
    /// it to build one `AnthropicMessagesProvider` per entry.
    pub messages_providers: Vec<(String, String)>,
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
    /// Semantic completion tier. Only consulted in a `--features semantic` build.
    pub semantic_enabled: bool,
    /// Redis-backed cross-replica single-flight. Only honoured on the
    /// redis storage backend (see main.rs).
    pub cross_replica_coalescing: bool,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub semantic_threshold: f32,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub semantic_match_unit: MatchUnit,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub semantic_index_maxlen: usize,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    pub semantic_poll_ms: u64,
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
            chat_providers: parse_chat_providers(
                std::env::var("ZEROCACHE_CHAT_PROVIDERS").ok().as_deref(),
            ),
            messages_providers: parse_messages_providers(
                std::env::var("ZEROCACHE_MESSAGES_PROVIDERS")
                    .ok()
                    .as_deref(),
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
            semantic_enabled: parse_semantic_enabled(
                std::env::var("ZEROCACHE_SEMANTIC").ok().as_deref(),
            ),
            cross_replica_coalescing: parse_cross_replica_coalescing(
                std::env::var("ZEROCACHE_CROSS_REPLICA_COALESCING")
                    .ok()
                    .as_deref(),
            ),
            semantic_threshold: parse_semantic_threshold(
                std::env::var("ZEROCACHE_SEMANTIC_THRESHOLD")
                    .ok()
                    .as_deref(),
            ),
            semantic_match_unit: parse_semantic_match_unit(
                std::env::var("ZEROCACHE_SEMANTIC_MATCH_UNIT")
                    .ok()
                    .as_deref(),
            ),
            semantic_index_maxlen: parse_semantic_index_maxlen(
                std::env::var("ZEROCACHE_SEMANTIC_INDEX_MAXLEN")
                    .ok()
                    .as_deref(),
            ),
            semantic_poll_ms: parse_semantic_poll_ms(
                std::env::var("ZEROCACHE_SEMANTIC_POLL_MS").ok().as_deref(),
            ),
        }
    }
}

fn parse_semantic_enabled(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("true") | Some("yes"))
}

fn parse_cross_replica_coalescing(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("true") | Some("yes"))
}

fn parse_semantic_threshold(raw: Option<&str>) -> f32 {
    match raw {
        None | Some("") => DEFAULT_SEMANTIC_THRESHOLD,
        Some(v) => match v.parse::<f32>() {
            Ok(f) if (SEMANTIC_THRESHOLD_FLOOR..=1.0).contains(&f) => f,
            _ => {
                eprintln!(
                    "warning: ZEROCACHE_SEMANTIC_THRESHOLD='{v}' is not a number in [{SEMANTIC_THRESHOLD_FLOOR}, 1.0] -- using {DEFAULT_SEMANTIC_THRESHOLD}"
                );
                DEFAULT_SEMANTIC_THRESHOLD
            }
        },
    }
}

fn parse_semantic_match_unit(raw: Option<&str>) -> MatchUnit {
    match raw {
        None | Some("") | Some("last-user") => MatchUnit::LastUser,
        Some("system-and-last-user") => MatchUnit::SystemAndLastUser,
        Some("full-conversation") => MatchUnit::FullConversation,
        Some(other) => {
            eprintln!(
                "warning: ZEROCACHE_SEMANTIC_MATCH_UNIT='{other}' is not recognized (expected last-user | system-and-last-user | full-conversation) -- using 'last-user'"
            );
            MatchUnit::LastUser
        }
    }
}

fn parse_semantic_index_maxlen(raw: Option<&str>) -> usize {
    match raw {
        None | Some("") => DEFAULT_SEMANTIC_INDEX_MAXLEN,
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!(
                    "warning: ZEROCACHE_SEMANTIC_INDEX_MAXLEN='{v}' is not a positive integer -- using {DEFAULT_SEMANTIC_INDEX_MAXLEN}"
                );
                DEFAULT_SEMANTIC_INDEX_MAXLEN
            }
        },
    }
}

fn parse_semantic_poll_ms(raw: Option<&str>) -> u64 {
    match raw {
        None | Some("") => DEFAULT_SEMANTIC_POLL_MS,
        Some(v) => match v.parse::<u64>() {
            Ok(n) if (SEMANTIC_POLL_MS_MIN..=SEMANTIC_POLL_MS_MAX).contains(&n) => n,
            Ok(n) => {
                let c = n.clamp(SEMANTIC_POLL_MS_MIN, SEMANTIC_POLL_MS_MAX);
                eprintln!(
                    "warning: ZEROCACHE_SEMANTIC_POLL_MS={v} is outside [{SEMANTIC_POLL_MS_MIN}, {SEMANTIC_POLL_MS_MAX}] -- using {c}"
                );
                c
            }
            Err(_) => {
                eprintln!(
                    "warning: ZEROCACHE_SEMANTIC_POLL_MS='{v}' is not a positive integer -- using {DEFAULT_SEMANTIC_POLL_MS}"
                );
                DEFAULT_SEMANTIC_POLL_MS
            }
        },
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

/// A chat provider name is a URL path segment: `[a-z0-9][a-z0-9_-]*`.
fn chat_provider_name_is_valid(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Parses ZEROCACHE_CHAT_PROVIDERS ("name=url,name=url") layered on top of
/// BUILTIN_CHAT_PROVIDERS. Each valid entry overrides a built-in's URL or
/// adds a new provider. A malformed entry is skipped with a warning; the
/// server always boots with at least the built-ins. Pure (takes the raw
/// value) so it is unit-testable without mutating real process env vars --
/// same pattern as parse_ttl_seconds.
fn parse_chat_providers(raw: Option<&str>) -> Vec<(String, String)> {
    let mut merged: Vec<(String, String)> = BUILTIN_CHAT_PROVIDERS
        .iter()
        .map(|(n, u)| (n.to_string(), normalize_chat_url(u)))
        .collect();

    let raw = match raw {
        Some(v) if !v.trim().is_empty() => v,
        _ => return merged,
    };

    for piece in raw.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (name, url) = match piece.split_once('=') {
            Some(pair) => pair,
            None => {
                eprintln!(
                    "warning: ZEROCACHE_CHAT_PROVIDERS entry '{piece}' has no '=' -- skipping"
                );
                continue;
            }
        };
        let name = name.trim().to_ascii_lowercase();
        let url = normalize_chat_url(url);
        if !chat_provider_name_is_valid(&name) {
            eprintln!(
                "warning: ZEROCACHE_CHAT_PROVIDERS name '{name}' is not a valid provider name (expected [a-z0-9][a-z0-9_-]*) -- skipping"
            );
            continue;
        }
        if url.is_empty() {
            eprintln!(
                "warning: ZEROCACHE_CHAT_PROVIDERS entry for '{name}' has an empty URL -- skipping"
            );
            continue;
        }
        match merged.iter_mut().find(|(n, _)| *n == name) {
            Some(entry) => entry.1 = url,
            None => merged.push((name, url)),
        }
    }

    merged
}

/// Normalizes a Messages-provider URL to the bare origin the Anthropic
/// adapter appends `/v1/messages` to. Forgiving: a bare origin, a trailing
/// slash, or a full `/v1/messages` (or `/v1`) suffix pasted from docs all
/// collapse to the same value. Not validated as a URL (same posture as the
/// ZEROCACHE_*_BASE_URL values).
fn normalize_messages_url(raw: &str) -> String {
    let mut s: &str = raw.trim().trim_end_matches('/');
    if let Some(stripped) = s.strip_suffix("/v1/messages") {
        s = stripped.trim_end_matches('/');
    } else if let Some(stripped) = s.strip_suffix("/v1") {
        s = stripped.trim_end_matches('/');
    }
    s.to_string()
}

/// Parses ZEROCACHE_MESSAGES_PROVIDERS ("name=url,name=url") layered on top of
/// BUILTIN_MESSAGES_PROVIDERS. A valid entry overrides a built-in's URL or
/// adds a new provider; a malformed entry is skipped with a warning and never
/// blocks boot. Pure (takes the raw value) for unit-testability — same
/// pattern as parse_chat_providers.
fn parse_messages_providers(raw: Option<&str>) -> Vec<(String, String)> {
    let mut merged: Vec<(String, String)> = BUILTIN_MESSAGES_PROVIDERS
        .iter()
        .map(|(n, u)| (n.to_string(), normalize_messages_url(u)))
        .collect();

    let raw = match raw {
        Some(v) if !v.trim().is_empty() => v,
        _ => return merged,
    };

    for piece in raw.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (name, url) = match piece.split_once('=') {
            Some(pair) => pair,
            None => {
                eprintln!(
                    "warning: ZEROCACHE_MESSAGES_PROVIDERS entry '{piece}' has no '=' -- skipping"
                );
                continue;
            }
        };
        let name = name.trim().to_ascii_lowercase();
        let url = normalize_messages_url(url);
        if !chat_provider_name_is_valid(&name) {
            eprintln!(
                "warning: ZEROCACHE_MESSAGES_PROVIDERS name '{name}' is not a valid provider name (expected [a-z0-9][a-z0-9_-]*) -- skipping"
            );
            continue;
        }
        if url.is_empty() {
            eprintln!(
                "warning: ZEROCACHE_MESSAGES_PROVIDERS entry for '{name}' has an empty URL -- skipping"
            );
            continue;
        }
        match merged.iter_mut().find(|(n, _)| *n == name) {
            Some(entry) => entry.1 = url,
            None => merged.push((name, url)),
        }
    }

    merged
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

    fn find<'a>(list: &'a [(String, String)], name: &str) -> Option<&'a str> {
        list.iter()
            .find(|(n, _)| n == name)
            .map(|(_, u)| u.as_str())
    }

    #[test]
    fn chat_providers_unset_yields_exactly_the_builtins() {
        let list = parse_chat_providers(None);
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len());
        assert_eq!(find(&list, "openai"), Some("https://api.openai.com/v1"));
        assert_eq!(
            find(&list, "gemini"),
            Some("https://generativelanguage.googleapis.com/v1beta/openai")
        );
        assert_eq!(find(&list, "groq"), Some("https://api.groq.com/openai/v1"));
    }

    #[test]
    fn chat_providers_empty_or_blank_is_treated_as_unset() {
        assert_eq!(
            parse_chat_providers(Some("")).len(),
            BUILTIN_CHAT_PROVIDERS.len()
        );
        assert_eq!(
            parse_chat_providers(Some("   ")).len(),
            BUILTIN_CHAT_PROVIDERS.len()
        );
    }

    #[test]
    fn chat_providers_override_replaces_a_builtin_url_only() {
        let list = parse_chat_providers(Some("openai=http://localhost:8000/v1"));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len());
        assert_eq!(find(&list, "openai"), Some("http://localhost:8000/v1"));
        assert_eq!(
            find(&list, "gemini"),
            Some("https://generativelanguage.googleapis.com/v1beta/openai")
        );
    }

    #[test]
    fn chat_providers_adds_a_new_name() {
        let list = parse_chat_providers(Some("ollama=http://localhost:11434/v1"));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len() + 1);
        assert_eq!(find(&list, "ollama"), Some("http://localhost:11434/v1"));
    }

    #[test]
    fn chat_providers_applies_multiple_comma_separated_entries() {
        let list = parse_chat_providers(Some(
            "ollama=http://localhost:11434/v1,acme=https://llm.acme.internal/v1",
        ));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len() + 2);
        assert_eq!(find(&list, "ollama"), Some("http://localhost:11434/v1"));
        assert_eq!(find(&list, "acme"), Some("https://llm.acme.internal/v1"));
    }

    #[test]
    fn chat_providers_tolerates_whitespace_around_name_and_url() {
        let list = parse_chat_providers(Some("  ollama =  http://localhost:11434/v1/  "));
        assert_eq!(find(&list, "ollama"), Some("http://localhost:11434/v1"));
    }

    #[test]
    fn chat_providers_normalizes_a_pasted_completions_url() {
        let list =
            parse_chat_providers(Some("groq=https://api.groq.com/openai/v1/chat/completions"));
        assert_eq!(find(&list, "groq"), Some("https://api.groq.com/openai/v1"));
    }

    #[test]
    fn chat_providers_skips_an_entry_with_no_equals_and_keeps_the_rest() {
        let list = parse_chat_providers(Some("garbage,ollama=http://x/v1"));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len() + 1);
        assert_eq!(find(&list, "ollama"), Some("http://x/v1"));
    }

    #[test]
    fn chat_providers_skips_an_illegal_name() {
        let list = parse_chat_providers(Some("Bad Name=http://x/v1"));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len());
        assert_eq!(find(&list, "bad name"), None);
    }

    #[test]
    fn chat_providers_skips_an_empty_url() {
        let list = parse_chat_providers(Some("ollama="));
        assert_eq!(list.len(), BUILTIN_CHAT_PROVIDERS.len());
        assert_eq!(find(&list, "ollama"), None);
    }

    #[test]
    fn chat_providers_preserves_an_equals_sign_inside_a_url() {
        let list = parse_chat_providers(Some("ollama=http://x/v1?token=abc"));
        assert_eq!(find(&list, "ollama"), Some("http://x/v1?token=abc"));
    }

    #[test]
    fn semantic_enabled_only_for_truthy_values() {
        assert!(parse_semantic_enabled(Some("1")));
        assert!(parse_semantic_enabled(Some("true")));
        assert!(parse_semantic_enabled(Some("yes")));
        assert!(!parse_semantic_enabled(Some("0")));
        assert!(!parse_semantic_enabled(Some("")));
        assert!(!parse_semantic_enabled(None));
    }

    #[test]
    fn semantic_threshold_defaults_and_validates() {
        assert_eq!(parse_semantic_threshold(None), DEFAULT_SEMANTIC_THRESHOLD);
        assert_eq!(
            parse_semantic_threshold(Some("")),
            DEFAULT_SEMANTIC_THRESHOLD
        );
        assert_eq!(parse_semantic_threshold(Some("0.9")), 0.9);
        assert_eq!(parse_semantic_threshold(Some("1.0")), 1.0);
        assert_eq!(
            parse_semantic_threshold(Some("0.2")),
            DEFAULT_SEMANTIC_THRESHOLD
        );
        assert_eq!(
            parse_semantic_threshold(Some("1.5")),
            DEFAULT_SEMANTIC_THRESHOLD
        );
        assert_eq!(
            parse_semantic_threshold(Some("abc")),
            DEFAULT_SEMANTIC_THRESHOLD
        );
    }

    #[test]
    fn semantic_match_unit_parses_each_name_and_defaults_on_unknown() {
        assert_eq!(parse_semantic_match_unit(None), MatchUnit::LastUser);
        assert_eq!(parse_semantic_match_unit(Some("")), MatchUnit::LastUser);
        assert_eq!(
            parse_semantic_match_unit(Some("last-user")),
            MatchUnit::LastUser
        );
        assert_eq!(
            parse_semantic_match_unit(Some("system-and-last-user")),
            MatchUnit::SystemAndLastUser
        );
        assert_eq!(
            parse_semantic_match_unit(Some("full-conversation")),
            MatchUnit::FullConversation
        );
        assert_eq!(
            parse_semantic_match_unit(Some("nonsense")),
            MatchUnit::LastUser
        );
    }

    #[test]
    fn builtin_chat_provider_urls_are_already_normalized() {
        for (name, url) in BUILTIN_CHAT_PROVIDERS {
            assert!(!url.is_empty(), "{name} has an empty URL");
            assert_eq!(
                &normalize_chat_url(url),
                url,
                "{name}'s built-in URL is not in normalized form"
            );
            assert!(!url.ends_with('/'), "{name} URL has a trailing slash");
            assert!(
                !url.ends_with("/chat/completions"),
                "{name} URL includes the /chat/completions suffix"
            );
        }
    }

    #[test]
    fn semantic_index_maxlen_defaults_and_validates() {
        assert_eq!(
            parse_semantic_index_maxlen(None),
            DEFAULT_SEMANTIC_INDEX_MAXLEN
        );
        assert_eq!(
            parse_semantic_index_maxlen(Some("")),
            DEFAULT_SEMANTIC_INDEX_MAXLEN
        );
        assert_eq!(parse_semantic_index_maxlen(Some("250000")), 250_000);
        assert_eq!(
            parse_semantic_index_maxlen(Some("0")),
            DEFAULT_SEMANTIC_INDEX_MAXLEN
        );
        assert_eq!(
            parse_semantic_index_maxlen(Some("-5")),
            DEFAULT_SEMANTIC_INDEX_MAXLEN
        );
        assert_eq!(
            parse_semantic_index_maxlen(Some("abc")),
            DEFAULT_SEMANTIC_INDEX_MAXLEN
        );
    }

    #[test]
    fn semantic_poll_ms_defaults_and_clamps() {
        assert_eq!(parse_semantic_poll_ms(None), DEFAULT_SEMANTIC_POLL_MS);
        assert_eq!(parse_semantic_poll_ms(Some("")), DEFAULT_SEMANTIC_POLL_MS);
        assert_eq!(parse_semantic_poll_ms(Some("1000")), 1000);
        assert_eq!(parse_semantic_poll_ms(Some("50")), 250);
        assert_eq!(parse_semantic_poll_ms(Some("120000")), 60_000);
        assert_eq!(
            parse_semantic_poll_ms(Some("abc")),
            DEFAULT_SEMANTIC_POLL_MS
        );
    }

    #[test]
    fn cross_replica_coalescing_only_for_truthy_values() {
        assert!(parse_cross_replica_coalescing(Some("1")));
        assert!(parse_cross_replica_coalescing(Some("true")));
        assert!(parse_cross_replica_coalescing(Some("yes")));
        assert!(!parse_cross_replica_coalescing(Some("0")));
        assert!(!parse_cross_replica_coalescing(Some("")));
        assert!(!parse_cross_replica_coalescing(Some("on")));
        assert!(!parse_cross_replica_coalescing(None));
    }

    fn find_msg<'a>(list: &'a [(String, String)], name: &str) -> Option<&'a str> {
        list.iter()
            .find(|(n, _)| n == name)
            .map(|(_, u)| u.as_str())
    }

    #[test]
    fn messages_providers_unset_yields_exactly_the_builtin() {
        let list = parse_messages_providers(None);
        assert_eq!(list.len(), BUILTIN_MESSAGES_PROVIDERS.len());
        assert_eq!(list.len(), 1);
        assert_eq!(
            find_msg(&list, "anthropic"),
            Some("https://api.anthropic.com")
        );
    }

    #[test]
    fn messages_providers_override_replaces_the_builtin_url() {
        let list = parse_messages_providers(Some("anthropic=https://gw.internal/anthropic"));
        assert_eq!(list.len(), 1);
        assert_eq!(
            find_msg(&list, "anthropic"),
            Some("https://gw.internal/anthropic")
        );
    }

    #[test]
    fn messages_providers_adds_a_new_name() {
        let list = parse_messages_providers(Some("selfhosted=https://llm.acme.internal"));
        assert_eq!(list.len(), 2);
        assert_eq!(
            find_msg(&list, "selfhosted"),
            Some("https://llm.acme.internal")
        );
    }

    #[test]
    fn messages_providers_skips_a_malformed_entry_and_keeps_the_rest() {
        let list = parse_messages_providers(Some("garbage,ok=https://x.example"));
        assert_eq!(list.len(), 2);
        assert_eq!(find_msg(&list, "ok"), Some("https://x.example"));
    }

    #[test]
    fn normalize_messages_url_trims_slash_v1_and_v1_messages() {
        assert_eq!(
            normalize_messages_url("https://api.anthropic.com/"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_messages_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_messages_url("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_messages_url("  https://api.anthropic.com/v1/messages/  "),
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn builtin_messages_provider_table_has_exactly_one_normalized_entry() {
        assert_eq!(BUILTIN_MESSAGES_PROVIDERS.len(), 1);
        for (name, url) in BUILTIN_MESSAGES_PROVIDERS {
            assert!(!url.is_empty(), "{name} has an empty URL");
            assert_eq!(
                &normalize_messages_url(url),
                url,
                "{name} URL not normalized"
            );
        }
    }
}

use std::time::Duration;

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
}

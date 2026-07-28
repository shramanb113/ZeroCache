use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};

/// A hung upstream connection must not block a request indefinitely. 30s is a
/// conservative ceiling for a same-region HTTPS call to a major cloud's
/// inference endpoint, not a measured SLA -- identical to the value the four
/// wire-shape-fixed adapters use, and uniform for the same reason: no
/// verified per-provider number exists to tune to.
pub const PROVIDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// reqwest-retry's DefaultRetryableStrategy retries 5xx / 408 / 429 /
/// connection errors and never other 4xx. That is exactly right for all three
/// clouds: Bedrock surfaces ThrottlingException and ModelNotReadyException as
/// 429, ModelTimeoutException as 408, and ServiceUnavailableException as 503,
/// while an expired credential is a 401/403 that retrying could only slow
/// down.
pub const MAX_RETRIES: u32 = 3;

/// The kit's own version, folded into every cloud adapter's `cache_scope`.
/// The per-cloud crate contributes its own CARGO_PKG_VERSION through
/// `EmbeddingProvider::version()`; this covers the case where the shared
/// driver's behavior changes without any cloud crate being bumped.
pub const KIT_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn build_client() -> ClientWithMiddleware {
    let inner = reqwest::Client::builder()
        .timeout(PROVIDER_TIMEOUT)
        .build()
        .expect("reqwest client with a timeout is always constructible");
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(MAX_RETRIES);
    ClientBuilder::new(inner)
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build()
}

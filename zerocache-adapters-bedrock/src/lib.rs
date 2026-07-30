//! Amazon Bedrock embedding adapter.
//!
//! Bedrock is one `InvokeModel` API in front of several independent vendors,
//! each with its own request/response JSON, so this crate is a
//! [`BedrockRouter`] plus one [`zerocache_adapters_cloud::TextWireStrategy`]
//! per vendor. All transport (client, retry, timeout, chunking, count
//! checking) comes from `zerocache-adapters-cloud`.
//!
//! Wire details verified live against AWS's own documentation on 2026-07-28:
//! `docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html`,
//! `.../userguide/api-keys-use.html`,
//! `.../userguide/model-parameters-titan-embed-text.html`,
//! `.../userguide/model-parameters-embed-v3.html`, and
//! `.../userguide/model-parameters-embed-v4.html`. That verification covers
//! only the documentation, not this code: the adapter itself is mock-only,
//! it has not yet had a live-key smoke test against a real AWS endpoint.
//!
//! Operational note (added after a live-docs cross-check, 2026-07-29/30):
//! Bedrock's short-term API keys are region-scoped (a key minted for one
//! region is not valid in another -- switching regions in the AWS console
//! mints a different key) and expire in up to 12 hours. Since `owner_id` is
//! derived from the caller's forwarded credential, a Bedrock caller's cache
//! namespace rotates whenever their key rotates, the same open risk
//! `zerocache-adapters-vertexai` already documents for its own OAuth2
//! access-token credential (see that crate's own module doc comment). A
//! caller whose `ZEROCACHE_BEDROCK_REGION` (or per-request region prefix)
//! doesn't match the region their key was minted for will see an auth
//! failure that looks like a Zerocache bug but originates entirely from AWS.

mod router;
mod strategy;

use zerocache_adapters_cloud::CloudProvider;

pub use router::{
    BedrockRouter, DEFAULT_BEDROCK_ENDPOINT_TEMPLATE, DEFAULT_BEDROCK_REGION, DEFAULT_COHERE_INPUT_TYPE,
};

pub type BedrockProvider = CloudProvider<BedrockRouter>;

/// Builds the adapter zerocache-http registers under the `bedrock` path
/// segment. `version` is this crate's own `CARGO_PKG_VERSION`, keeping
/// cache-key versioning tied to this Cargo.toml rather than the kit's.
pub fn new_provider(default_region: impl Into<String>, endpoint_template: impl Into<String>) -> BedrockProvider {
    CloudProvider::new(
        BedrockRouter::new(default_region, endpoint_template),
        env!("CARGO_PKG_VERSION"),
    )
}

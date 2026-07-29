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

//! Azure embedding adapter, covering both of Azure's embedding surfaces.
//!
//! Azure OpenAI hosts OpenAI's models only; the non-OpenAI embedding models
//! (Cohere and friends) live on Microsoft Foundry Models, on a different
//! hostname with a slightly different request contract. Both are reachable
//! here, selected by a `foundry:` prefix on the caller's `model` string. All
//! transport (client, retry, timeout, chunking, count checking) comes from
//! `zerocache-adapters-cloud`.
//!
//! `model` is an Azure *deployment name*, not an upstream model name -- two
//! resources can name two different models identically, which is why the
//! resource hostname is part of the cache scope.
//!
//! Wire details verified live against Microsoft's own documentation on
//! 2026-07-28:
//! `learn.microsoft.com/en-us/azure/foundry/openai/how-to/embeddings` and
//! `learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/how-to/use-embeddings`.
//! The latter carries a retirement notice for the Azure AI Inference beta SDK
//! (2026-08-26) pointing callers at `/openai/v1`, which is why that surface is
//! the default here and Foundry is opt-in. That verification covers only the
//! documentation, not this code: the adapter itself is mock-only, it has not
//! yet had a live-key smoke test against a real Azure endpoint.

mod router;
mod strategy;

use zerocache_adapters_cloud::CloudProvider;

pub use router::{
    AzureAuthMode, AzureRouter, DEFAULT_AZURE_FOUNDRY_API_VERSION, FOUNDRY_MODEL_PREFIX,
};

pub type AzureProvider = CloudProvider<AzureRouter>;

/// Builds the adapter zerocache-http registers under the `azure` path
/// segment. `version` is this crate's own `CARGO_PKG_VERSION`, keeping
/// cache-key versioning tied to this Cargo.toml rather than the kit's.
pub fn new_provider(
    openai_base_url: Option<String>,
    foundry_base_url: Option<String>,
    foundry_api_version: impl Into<String>,
    auth_mode: AzureAuthMode,
) -> AzureProvider {
    CloudProvider::new(
        AzureRouter::new(openai_base_url, foundry_base_url, foundry_api_version, auth_mode),
        env!("CARGO_PKG_VERSION"),
    )
}

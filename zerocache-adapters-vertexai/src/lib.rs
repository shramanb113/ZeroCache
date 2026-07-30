//! GCP Vertex AI embedding adapter.
//!
//! Vertex has no OpenAI-compatible embeddings endpoint -- its OpenAI-compat
//! surface is chat-completions only -- so this crate implements the native
//! `:predict` wire shape. All transport (client, retry, timeout, chunking,
//! count checking) comes from `zerocache-adapters-cloud`.
//!
//! Auth is the caller's own OAuth2 access token, forwarded as a bearer token.
//! Zerocache holds no service account. Note that a GCP access token lives
//! roughly an hour, and `owner_id` is derived from the credential, so a
//! caller's cache namespace rotates with their token -- see the design spec's
//! "Open risk: Vertex token rotation vs. owner_id".
//!
//! Wire details verified live against Google's own documentation on
//! 2026-07-28:
//! `docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings`,
//! `.../docs/model-reference/text-embeddings-api`, and
//! `.../vertex-ai/docs/reference/rest/v1beta1/projects.locations.endpoints.chat/completions`
//! (which is what establishes that the OpenAI-compat surface excludes
//! embeddings). That verification covers only the documentation, not this
//! code: the adapter itself is mock-only, it has not yet had a live-key
//! smoke test against a real GCP endpoint.
//!
//! Re-verified 2026-07-29/30 during a dedicated live-docs cross-check (no
//! API keys are available for this project, so this kind of periodic
//! doc-vs-code verification substitutes for a live smoke test). Google's
//! docs site has since been rebranded from `vertex-ai/generative-ai/docs/...`
//! to `gemini-enterprise-agent-platform/...` -- the underlying API is
//! unchanged, only the docs URLs moved. Current citations:
//! `docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings`,
//! `.../reference/models/text-embeddings-api`, `.../resources/locations`,
//! `.../models/model-versions`. This pass found and fixed two real defects:
//! the `global`/`us`/`eu` location values previously produced a wrong,
//! nonexistent endpoint host (empirically confirmed via unauthenticated
//! probes), and `gemini-embedding-001`'s batch limit was stale at 1 when
//! current docs apply the shared 250-instance limit. See `router.rs`'s
//! `resolve_host` and `GEMINI_MAX_BATCH` for details.

mod router;
mod strategy;

use zerocache_adapters_cloud::CloudProvider;

pub use router::{VertexRouter, DEFAULT_VERTEX_ENDPOINT_TEMPLATE, DEFAULT_VERTEX_LOCATION};

pub type VertexProvider = CloudProvider<VertexRouter>;

/// Builds the adapter zerocache-http registers under the `vertexai` path
/// segment. `version` is this crate's own `CARGO_PKG_VERSION`, keeping
/// cache-key versioning tied to this Cargo.toml rather than the kit's.
pub fn new_provider(
    default_project: Option<String>,
    default_location: impl Into<String>,
    endpoint_template: impl Into<String>,
) -> VertexProvider {
    CloudProvider::new(
        VertexRouter::new(default_project, default_location, endpoint_template),
        env!("CARGO_PKG_VERSION"),
    )
}

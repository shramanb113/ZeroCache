use zerocache_ports::{ProviderError, ProviderUsage};

/// One caller-supplied `model` string resolved into concrete coordinates for
/// one cloud.
///
/// Every field is derived per request, which is what makes per-request
/// variation (Bedrock region, Vertex project, Azure surface, Cohere
/// input_type) visible to the cache key instead of silently collapsing into
/// one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Fully-qualified, normalized form. Two spellings of the same target
    /// must produce the same `canonical`; two different targets must never
    /// produce the same one. This is what lands in `cache_scope`.
    pub canonical: String,
    /// The bare model/deployment identifier the wire body or URL path needs.
    pub model_id: String,
    /// Scheme + host (+ any fixed prefix) for this specific request.
    pub endpoint_base: String,
    /// Cloud-specific extra the strategy needs and the driver never inspects
    /// -- Bedrock's `input_type`, Vertex's `task_type`, Azure's surface.
    pub qualifier: Option<String>,
}

/// Everything one upstream HTTP call needs.
///
/// `body` is pre-serialized bytes rather than a `serde_json::Value` so each
/// strategy keeps its own typed request structs, matching the style of the
/// four existing adapters, instead of hand-assembling untyped JSON.
pub struct EmbedCall {
    pub url: String,
    /// Sent in addition to `Content-Type: application/json`, which the driver
    /// always sets.
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

pub struct EmbedOutcome {
    pub vectors: Vec<Vec<f32>>,
    pub usage: ProviderUsage,
}

/// One vendor's wire shape within one cloud. Stateless -- everything
/// per-request arrives as an argument -- so a single instance is shared
/// across every request the process serves.
pub trait TextWireStrategy: Send + Sync {
    /// Inputs per upstream call. A property of the strategy, not a crate-wide
    /// constant, because the real limits differ by an order of magnitude and
    /// two of them are exactly 1: Bedrock's Titan takes a scalar `inputText`,
    /// and Vertex's gemini-embedding-001 accepts a single instance, while
    /// Bedrock Cohere takes 96 and Vertex text-embedding-005 takes 250.
    fn max_batch(&self) -> usize;

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError>;

    /// `expected` is the chunk length, for strategies whose wire shape needs
    /// it to parse. The driver enforces the returned count independently, so
    /// a strategy that does not need it may ignore it.
    fn parse_response(&self, expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError>;
}

/// One cloud's routing table: operator configuration plus the mapping from a
/// caller's `model` string to a concrete endpoint and wire strategy.
pub trait CloudRouter: Send + Sync {
    fn resolve(&self, model: &str) -> Result<ResolvedModel, ProviderError>;
    fn strategy_for(&self, resolved: &ResolvedModel) -> Result<&dyn TextWireStrategy, ProviderError>;
}

use serde::{Deserialize, Serialize};
use zerocache_adapters_cloud::{EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};
use zerocache_ports::{ProviderError, ProviderUsage};

use crate::router::AzureAuthMode;

/// Conservative and uniform with every other adapter in this workspace, not a
/// measured limit. Azure allows up to 2,048 array items but caps a request at
/// 300,000 aggregate tokens with a 400; Foundry's documented default is around
/// 1,024. 100 stays clear of all of them.
const MAX_BATCH: usize = 100;

// ------------------------------------------------- shared response shape ----
//
// Both Azure surfaces return OpenAI's exact envelope, including the per-item
// `index` that lets a response come back out of order. Parsing is therefore
// written once.

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<UsageResponse>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct UsageResponse {
    prompt_tokens: u32,
    total_tokens: u32,
}

fn parse_openai_shaped(expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
    let parsed: EmbeddingsResponse =
        serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;

    if parsed.data.len() != expected {
        // Reported here rather than left to the driver so the index-bounds
        // check below can't panic on a short-but-high-index response.
        return Err(ProviderError(format!(
            "expected {expected} embeddings in response, got {}",
            parsed.data.len()
        )));
    }

    let mut ordered = vec![Vec::new(); expected];
    for item in parsed.data {
        if item.index >= expected {
            return Err(ProviderError(format!(
                "response embedding index {} is out of range for a batch of {expected}",
                item.index
            )));
        }
        ordered[item.index] = item.embedding;
    }

    let usage = parsed
        .usage
        .map(|u| ProviderUsage { prompt_tokens: u.prompt_tokens, total_tokens: u.total_tokens })
        .unwrap_or_default();

    Ok(EmbedOutcome { vectors: ordered, usage })
}

// ------------------------------------------------------- Azure OpenAI v1 ----

#[derive(Serialize)]
struct OpenAiV1Request<'a> {
    model: &'a str,
    input: &'a [String],
}

/// `POST {resource}/openai/v1/embeddings` -- Azure's GA v1 API. No
/// `api-version` query parameter on the GA path. `model` is the deployment
/// name.
pub struct AzureOpenAiV1Strategy {
    auth_mode: AzureAuthMode,
}

impl AzureOpenAiV1Strategy {
    pub fn new(auth_mode: AzureAuthMode) -> Self {
        Self { auth_mode }
    }
}

impl TextWireStrategy for AzureOpenAiV1Strategy {
    fn max_batch(&self) -> usize {
        MAX_BATCH
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let body = serde_json::to_vec(&OpenAiV1Request { model: &resolved.model_id, input: texts })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall {
            url: resolved.endpoint_base.clone(),
            headers: vec![self.auth_mode.header(api_key)],
            body,
        })
    }

    fn parse_response(&self, expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        parse_openai_shaped(expected, body)
    }
}

// ------------------------------------------------------- Foundry Models ----

#[derive(Serialize)]
struct FoundryRequest<'a> {
    model: &'a str,
    input: &'a [String],
    /// Foundry-only. Not every Foundry model accepts it -- those return 422,
    /// which is a fatal 4xx the retry policy correctly does not retry -- so it
    /// is only sent when the caller explicitly asked for one.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
}

/// `POST {resource}/models/embeddings?api-version=…` -- the surface carrying
/// the non-OpenAI embedding vendors (Cohere and friends).
pub struct AzureFoundryStrategy {
    auth_mode: AzureAuthMode,
}

impl AzureFoundryStrategy {
    pub fn new(auth_mode: AzureAuthMode) -> Self {
        Self { auth_mode }
    }
}

impl TextWireStrategy for AzureFoundryStrategy {
    fn max_batch(&self) -> usize {
        MAX_BATCH
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let body = serde_json::to_vec(&FoundryRequest {
            model: &resolved.model_id,
            input: texts,
            input_type: resolved.qualifier.as_deref(),
        })
        .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall {
            url: resolved.endpoint_base.clone(),
            headers: vec![self.auth_mode.header(api_key)],
            body,
        })
    }

    fn parse_response(&self, expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        parse_openai_shaped(expected, body)
    }
}

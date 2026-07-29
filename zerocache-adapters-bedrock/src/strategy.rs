use serde::{Deserialize, Serialize};
use zerocache_adapters_cloud::{EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};
use zerocache_ports::{ProviderError, ProviderUsage};

/// Bedrock's API-key auth: a plain bearer token against bedrock-runtime, no
/// SigV4. Verified 2026-07-28 against AWS's own cURL example in
/// docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html.
fn bedrock_headers(api_key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("Authorization", format!("Bearer {api_key}")),
        ("Accept", "application/json".to_string()),
    ]
}

fn invoke_url(resolved: &ResolvedModel) -> String {
    format!("{}/model/{}/invoke", resolved.endpoint_base, resolved.model_id)
}

// ---------------------------------------------------------------- Titan ----

#[derive(Serialize)]
struct TitanRequest<'a> {
    #[serde(rename = "inputText")]
    input_text: &'a str,
}

#[derive(Deserialize)]
struct TitanResponse {
    embedding: Vec<f32>,
    #[serde(rename = "inputTextTokenCount")]
    input_text_token_count: u32,
}

/// Amazon Titan Embeddings (`amazon.titan-embed-text-v1`,
/// `amazon.titan-embed-text-v2:0`).
///
/// `inputText` is a scalar string, not an array, so one text per call is not a
/// conservative choice -- it is the only thing the API accepts. A 500-chunk
/// ingestion batch that misses entirely costs 500 sequential HTTP calls
/// against Titan, which is worth knowing before choosing it as an ingestion
/// model.
pub struct TitanEmbedStrategy;

impl TextWireStrategy for TitanEmbedStrategy {
    fn max_batch(&self) -> usize {
        1
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let text = texts.first().ok_or_else(|| {
            ProviderError("titan strategy called with an empty chunk".to_string())
        })?;

        // Deliberately not sending `dimensions`, `normalize`, or
        // `embeddingTypes`: all three change the returned vector, none is
        // expressible in the OpenAI-shaped wire contract Zerocache exposes,
        // and none would be visible in the cache key. Titan's own defaults
        // (1024 dims, normalized, float) are what a caller gets.
        let body = serde_json::to_vec(&TitanRequest { input_text: text })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall { url: invoke_url(resolved), headers: bedrock_headers(api_key), body })
    }

    fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        let parsed: TitanResponse =
            serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;
        Ok(EmbedOutcome {
            vectors: vec![parsed.embedding],
            usage: ProviderUsage {
                prompt_tokens: parsed.input_text_token_count,
                total_tokens: parsed.input_text_token_count,
            },
        })
    }
}

// --------------------------------------------------------------- Cohere ----

#[derive(Serialize)]
struct CohereRequest<'a> {
    texts: &'a [String],
    input_type: &'a str,
}

/// Cohere's response is polymorphic: a flat `[[f32]]` when `embedding_types`
/// was not requested (`response_type: "embeddings_floats"`), or a
/// type-keyed object when it was (`response_type: "embeddings_by_type"`).
/// This strategy never requests `embedding_types`, so the flat form is what
/// it expects -- but it parses both, so a Bedrock-side default change
/// degrades into a working parse instead of a deserialization error at
/// runtime.
#[derive(Deserialize)]
#[serde(untagged)]
enum CohereEmbeddings {
    Flat(Vec<Vec<f32>>),
    ByType { float: Vec<Vec<f32>> },
}

#[derive(Deserialize)]
struct CohereResponse {
    embeddings: CohereEmbeddings,
}

/// Cohere Embed v3 (`cohere.embed-english-v3`,
/// `cohere.embed-multilingual-v3`) and v4 (`cohere.embed-v4:0`). Both accept
/// `texts` + `input_type` and return the same envelope, so one strategy
/// covers both.
pub struct CohereEmbedStrategy;

impl TextWireStrategy for CohereEmbedStrategy {
    fn max_batch(&self) -> usize {
        // AWS documents "0 to 96 texts per call" for v3 and "Max 96 per call"
        // for v4 -- the same number, and a real API limit rather than the
        // unverified 100 the four wire-shape-fixed adapters share.
        96
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let input_type = resolved.qualifier.as_deref().ok_or_else(|| {
            ProviderError("cohere strategy reached without a resolved input_type".to_string())
        })?;

        let body = serde_json::to_vec(&CohereRequest { texts, input_type })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall { url: invoke_url(resolved), headers: bedrock_headers(api_key), body })
    }

    fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        let parsed: CohereResponse =
            serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;
        let vectors = match parsed.embeddings {
            CohereEmbeddings::Flat(v) => v,
            CohereEmbeddings::ByType { float } => float,
        };
        // Cohere on Bedrock reports no token usage at all -- report zero
        // rather than fabricate, the same posture as Gemini and HuggingFace.
        Ok(EmbedOutcome { vectors, usage: ProviderUsage::default() })
    }
}

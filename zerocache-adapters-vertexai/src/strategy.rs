use serde::{Deserialize, Serialize};
use zerocache_adapters_cloud::{EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};
use zerocache_ports::{ProviderError, ProviderUsage};

// NOTE ON CASING: Vertex's :predict payload mixes conventions and this is not
// a mistake in the structs below. Instance fields are snake_case
// (`task_type`, `content`), the `parameters` object is camelCase
// (`autoTruncate`, `outputDimensionality`), and `statistics` is snake_case
// again (`token_count`). Verified 2026-07-28. Do not replace the per-field
// renames with a blanket rename_all.

#[derive(Serialize)]
struct PredictRequest {
    instances: Vec<Instance>,
}

#[derive(Serialize)]
struct Instance {
    content: String,
    /// Omitted entirely when the caller did not supply a `#<task_type>`
    /// qualifier, so Google applies its own default rather than this adapter
    /// inventing one.
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
}

#[derive(Deserialize)]
struct PredictResponse {
    predictions: Vec<Prediction>,
}

#[derive(Deserialize)]
struct Prediction {
    embeddings: Embeddings,
}

#[derive(Deserialize)]
struct Embeddings {
    values: Vec<f32>,
    #[serde(default)]
    statistics: Option<Statistics>,
}

#[derive(Deserialize)]
struct Statistics {
    #[serde(rename = "token_count")]
    token_count: u32,
}

/// The Vertex `:predict` wire shape. One type, instantiated at two batch
/// sizes: Google allows 250 instances per request for the text-embedding
/// models and exactly 1 for gemini-embedding-001.
pub struct VertexPredictStrategy {
    max_batch: usize,
}

impl VertexPredictStrategy {
    pub fn new(max_batch: usize) -> Self {
        Self { max_batch }
    }
}

impl TextWireStrategy for VertexPredictStrategy {
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn build_call(
        &self,
        api_key: &str,
        resolved: &ResolvedModel,
        texts: &[String],
    ) -> Result<EmbedCall, ProviderError> {
        let instances = texts
            .iter()
            .map(|text| Instance {
                content: text.clone(),
                task_type: resolved.qualifier.clone(),
            })
            .collect();

        // `parameters` is deliberately not sent: outputDimensionality and
        // autoTruncate both change the returned vector, neither is expressible
        // in the OpenAI-shaped wire contract Zerocache exposes, and neither
        // would be visible in the cache key. Google's own defaults apply.
        let body = serde_json::to_vec(&PredictRequest { instances })
            .map_err(|e| ProviderError(e.to_string()))?;

        Ok(EmbedCall {
            url: format!("{}/{}:predict", resolved.endpoint_base, resolved.model_id),
            headers: vec![("Authorization", format!("Bearer {api_key}"))],
            body,
        })
    }

    fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
        let parsed: PredictResponse =
            serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;

        let mut vectors = Vec::with_capacity(parsed.predictions.len());
        let mut tokens: u32 = 0;
        for prediction in parsed.predictions {
            if let Some(statistics) = &prediction.embeddings.statistics {
                tokens = tokens.saturating_add(statistics.token_count);
            }
            vectors.push(prediction.embeddings.values);
        }

        Ok(EmbedOutcome {
            vectors,
            // Vertex reports one token_count per prediction rather than a
            // request-level usage object; summing them is the closest honest
            // equivalent, and it is a real number, not a fabricated one.
            usage: ProviderUsage { prompt_tokens: tokens, total_tokens: tokens },
        })
    }
}

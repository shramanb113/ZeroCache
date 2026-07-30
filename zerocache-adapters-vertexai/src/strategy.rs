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

/// The Vertex `:predict` wire shape. One type, instantiated per model
/// family in `router.rs` (`STANDARD_MAX_BATCH`/`GEMINI_MAX_BATCH`) -- as of
/// 2026-07-29/30 both share Google's documented 250-instance-per-request
/// limit; there is no per-model carve-out any more (see `router.rs` for the
/// history of that constant).
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

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;
    use zerocache_ports::EmbeddingProvider;

    use crate::{new_provider, VertexProvider};

    fn provider(base: String) -> VertexProvider {
        new_provider(Some("my-project".to_string()), "us-central1", base)
    }

    const PREDICT_PATH: &str =
        "/v1/projects/my-project/locations/us-central1/publishers/google/models/text-embedding-005:predict";

    #[tokio::test]
    async fn sends_instances_with_bearer_auth_and_sums_token_count_as_usage() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(PREDICT_PATH)
                    .header("Authorization", "Bearer test-token")
                    .json_body(json!({ "instances": [{ "content": "a" }, { "content": "b" }] }));
                then.status(200).json_body(json!({
                    "predictions": [
                        { "embeddings": { "values": [1.0], "statistics": { "token_count": 3, "truncated": false } } },
                        { "embeddings": { "values": [2.0], "statistics": { "token_count": 4, "truncated": false } } }
                    ]
                }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url())
            .embed_batch("test-token", "text-embedding-005", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(usage.prompt_tokens, 7, "Vertex reports per-prediction token_count; it must be summed, not dropped");
        assert_eq!(usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn omits_task_type_entirely_when_the_caller_did_not_ask_for_one() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH).matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body["instances"][0].get("task_type").is_none()
                });
                then.status(200)
                    .json_body(json!({ "predictions": [{ "embeddings": { "values": [1.0] } }] }));
            })
            .await;

        provider(server.base_url())
            .embed_batch("test-token", "text-embedding-005", &["a".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn a_task_type_qualifier_reaches_every_instance_in_the_request() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH).json_body(json!({
                    "instances": [
                        { "content": "a", "task_type": "RETRIEVAL_DOCUMENT" },
                        { "content": "b", "task_type": "RETRIEVAL_DOCUMENT" }
                    ]
                }));
                then.status(200).json_body(json!({
                    "predictions": [
                        { "embeddings": { "values": [1.0] } },
                        { "embeddings": { "values": [2.0] } }
                    ]
                }));
            })
            .await;

        provider(server.base_url())
            .embed_batch(
                "test-token",
                "text-embedding-005#RETRIEVAL_DOCUMENT",
                &["a".to_string(), "b".to_string()],
            )
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn a_missing_statistics_block_reports_zero_usage_rather_than_failing() {
        // The `statistics` object is not guaranteed on every prediction; a
        // missing one must not turn a perfectly good vector into an error.
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH);
                then.status(200)
                    .json_body(json!({ "predictions": [{ "embeddings": { "values": [1.0, 2.0] } }] }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url())
            .embed_batch("test-token", "text-embedding-005", &["a".to_string()])
            .await
            .unwrap();

        assert_eq!(vectors, vec![vec![1.0, 2.0]]);
        assert_eq!(usage.prompt_tokens, 0);
    }

    #[tokio::test]
    async fn gemini_embedding_batches_multiple_texts_in_one_call() {
        // Superseded 2026-07-29/30: gemini-embedding-001 previously accepted
        // only one instance per request (see router.rs's GEMINI_MAX_BATCH
        // history), but current docs apply the same shared 250-instance
        // limit as the other Google embedding models, so a small batch like
        // this one now goes out as a single call, not one call per text.
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-embedding-001:predict")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                        body["instances"].as_array().map(|a| a.len()) == Some(3)
                    });
                then.status(200).json_body(json!({
                    "predictions": [
                        { "embeddings": { "values": [9.0], "statistics": { "token_count": 1 } } },
                        { "embeddings": { "values": [9.1], "statistics": { "token_count": 1 } } },
                        { "embeddings": { "values": [9.2], "statistics": { "token_count": 1 } } }
                    ]
                }));
            })
            .await;

        let texts: Vec<String> = (0..3).map(|i| format!("t{i}")).collect();
        let (vectors, usage) = provider(server.base_url())
            .embed_batch("test-token", "gemini-embedding-001", &texts)
            .await
            .unwrap();

        assert_eq!(mock.hits_async().await, 1, "gemini-embedding-001 now shares the 250-instance batch limit, so 3 texts fit in one call");
        assert_eq!(vectors.len(), 3);
        assert_eq!(usage.prompt_tokens, 3);
    }

    #[tokio::test]
    async fn chunks_at_two_hundred_fifty_and_concatenates_in_order() {
        let server = MockServer::start_async().await;
        let first = server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH).matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body["instances"].as_array().map(|a| a.len()) == Some(250)
                });
                then.status(200).json_body_obj(&json!({
                    "predictions": (0..250)
                        .map(|i| json!({ "embeddings": { "values": [i as f64] } }))
                        .collect::<Vec<_>>()
                }));
            })
            .await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH).matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body["instances"].as_array().map(|a| a.len()) == Some(10)
                });
                then.status(200).json_body_obj(&json!({
                    "predictions": (0..10)
                        .map(|i| json!({ "embeddings": { "values": [1000.0 + i as f64] } }))
                        .collect::<Vec<_>>()
                }));
            })
            .await;

        let texts: Vec<String> = (0..260).map(|i| format!("text-{i}")).collect();
        let (vectors, _usage) = provider(server.base_url())
            .embed_batch("test-token", "text-embedding-005", &texts)
            .await
            .unwrap();

        first.assert_async().await;
        second.assert_async().await;
        assert_eq!(vectors.len(), 260);
        assert_eq!(vectors[0], vec![0.0]);
        assert_eq!(vectors[249], vec![249.0]);
        assert_eq!(vectors[250], vec![1000.0]);
        assert_eq!(vectors[259], vec![1009.0]);
    }

    #[tokio::test]
    async fn an_http_error_status_surfaces_as_an_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH);
                then.status(401).json_body(json!({ "error": { "message": "invalid credentials" } }));
            })
            .await;

        let result = provider(server.base_url())
            .embed_batch("stale-token", "text-embedding-005", &["x".to_string()])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn a_prediction_count_mismatch_is_a_hard_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path(PREDICT_PATH);
                then.status(200)
                    .json_body(json!({ "predictions": [{ "embeddings": { "values": [1.0] } }] }));
            })
            .await;

        let result = provider(server.base_url())
            .embed_batch("test-token", "text-embedding-005", &["a".to_string(), "b".to_string()])
            .await;

        assert!(result.is_err(), "a count mismatch must be a hard error, not a silent misalignment");
    }

    #[test]
    fn cache_scope_separates_projects_locations_models_and_task_types() {
        let p = provider("https://{location}-aiplatform.googleapis.com".to_string());
        let base = p.cache_scope("text-embedding-005").unwrap();
        let other_project = p.cache_scope("us-central1/other-project/text-embedding-005").unwrap();
        let other_location = p.cache_scope("europe-west4/my-project/text-embedding-005").unwrap();
        let other_task = p.cache_scope("text-embedding-005#RETRIEVAL_QUERY").unwrap();
        let other_model = p.cache_scope("gemini-embedding-001").unwrap();

        assert_ne!(base, other_project);
        assert_ne!(base, other_location);
        assert_ne!(base, other_task);
        assert_ne!(base, other_model);
    }

    #[test]
    fn cache_scope_rejects_a_malformed_model_rather_than_inventing_a_scope() {
        let p = provider("https://{location}-aiplatform.googleapis.com".to_string());
        assert!(p.cache_scope("a/b/c/d").is_err());
    }
}

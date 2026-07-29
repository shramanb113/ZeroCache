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

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;
    use zerocache_ports::EmbeddingProvider;

    use crate::{new_provider, AzureAuthMode, AzureProvider};

    fn provider(base: String, auth_mode: AzureAuthMode) -> AzureProvider {
        new_provider(base.clone(), Some(base), "2024-05-01-preview", auth_mode)
    }

    #[tokio::test]
    async fn openai_v1_sends_the_deployment_as_model_and_reorders_by_index() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/openai/v1/embeddings")
                    .header("Authorization", "Bearer entra-token")
                    .json_body(json!({ "model": "my-deployment", "input": ["a", "b"] }));
                then.status(200).json_body(json!({
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        { "object": "embedding", "embedding": [2.0], "index": 1 },
                        { "object": "embedding", "embedding": [1.0], "index": 0 }
                    ],
                    "usage": { "prompt_tokens": 5, "total_tokens": 5 }
                }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("entra-token", "my-deployment", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]], "an out-of-order response must be reordered by index");
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn openai_v1_sends_no_api_version_query_parameter() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings").matches(|req| {
                    req.query_params
                        .as_ref()
                        .map(|params| params.iter().all(|(k, _)| k != "api-version"))
                        .unwrap_or(true)
                });
                then.status(200).json_body(json!({
                    "data": [{ "embedding": [1.0], "index": 0 }],
                    "usage": { "prompt_tokens": 1, "total_tokens": 1 }
                }));
            })
            .await;

        provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "my-deployment", &["a".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn api_key_auth_mode_sends_the_api_key_header_instead_of_a_bearer_token() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings").header("api-key", "resource-key");
                then.status(200).json_body(json!({
                    "data": [{ "embedding": [1.0], "index": 0 }],
                    "usage": { "prompt_tokens": 1, "total_tokens": 1 }
                }));
            })
            .await;

        provider(server.base_url(), AzureAuthMode::ApiKey)
            .embed_batch("resource-key", "my-deployment", &["a".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn foundry_sends_input_type_when_the_caller_asked_and_omits_it_otherwise() {
        let server = MockServer::start_async().await;
        let with_type = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/embeddings")
                    .query_param("api-version", "2024-05-01-preview")
                    .json_body(json!({
                        "model": "cohere-embed-v3-english",
                        "input": ["a"],
                        "input_type": "query"
                    }));
                then.status(200).json_body(json!({
                    "data": [{ "embedding": [1.0], "index": 0 }],
                    "usage": { "prompt_tokens": 2, "total_tokens": 2 }
                }));
            })
            .await;

        provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "foundry:cohere-embed-v3-english#query", &["a".to_string()])
            .await
            .unwrap();
        with_type.assert_async().await;

        let without_type = server
            .mock_async(|when, then| {
                when.method(POST).path("/models/embeddings").matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body.get("input_type").is_none()
                });
                then.status(200).json_body(json!({
                    "data": [{ "embedding": [3.0], "index": 0 }],
                    "usage": { "prompt_tokens": 2, "total_tokens": 2 }
                }));
            })
            .await;

        provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "foundry:cohere-embed-v3-english", &["a".to_string()])
            .await
            .unwrap();
        without_type.assert_async().await;
    }

    #[tokio::test]
    async fn a_missing_usage_object_reports_zero_rather_than_failing() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings");
                then.status(200).json_body(json!({ "data": [{ "embedding": [1.0], "index": 0 }] }));
            })
            .await;

        let (vectors, usage) = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "my-deployment", &["a".to_string()])
            .await
            .unwrap();

        assert_eq!(vectors, vec![vec![1.0]]);
        assert_eq!(usage.prompt_tokens, 0, "absent usage must be zero, not fabricated");
    }

    #[tokio::test]
    async fn chunks_at_one_hundred_and_concatenates_in_order() {
        let server = MockServer::start_async().await;
        let first = server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings").matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body["input"].as_array().map(|a| a.len()) == Some(100)
                });
                then.status(200).json_body_obj(&json!({
                    "data": (0..100)
                        .map(|i| json!({ "embedding": [i as f64], "index": i }))
                        .collect::<Vec<_>>(),
                    "usage": { "prompt_tokens": 100, "total_tokens": 100 }
                }));
            })
            .await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings").matches(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                    body["input"].as_array().map(|a| a.len()) == Some(20)
                });
                then.status(200).json_body_obj(&json!({
                    "data": (0..20)
                        .map(|i| json!({ "embedding": [1000.0 + i as f64], "index": i }))
                        .collect::<Vec<_>>(),
                    "usage": { "prompt_tokens": 20, "total_tokens": 20 }
                }));
            })
            .await;

        let texts: Vec<String> = (0..120).map(|i| format!("text-{i}")).collect();
        let (vectors, usage) = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "my-deployment", &texts)
            .await
            .unwrap();

        first.assert_async().await;
        second.assert_async().await;
        assert_eq!(vectors.len(), 120);
        assert_eq!(vectors[0], vec![0.0]);
        assert_eq!(vectors[99], vec![99.0]);
        assert_eq!(vectors[100], vec![1000.0]);
        assert_eq!(usage.prompt_tokens, 120, "usage must accumulate across chunks");
    }

    #[tokio::test]
    async fn an_http_error_status_surfaces_as_an_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings");
                then.status(401).json_body(json!({ "error": { "message": "invalid token" } }));
            })
            .await;

        let result = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("stale", "my-deployment", &["x".to_string()])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn a_response_count_mismatch_is_a_hard_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings");
                then.status(200).json_body(json!({ "data": [{ "embedding": [1.0], "index": 0 }] }));
            })
            .await;

        let result = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "my-deployment", &["a".to_string(), "b".to_string()])
            .await;

        assert!(result.is_err(), "a count mismatch must be a hard error, not a silent misalignment");
    }

    #[tokio::test]
    async fn an_out_of_range_index_is_an_error_not_a_panic() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/openai/v1/embeddings");
                then.status(200).json_body(json!({
                    "data": [
                        { "embedding": [1.0], "index": 0 },
                        { "embedding": [2.0], "index": 7 }
                    ]
                }));
            })
            .await;

        let result = provider(server.base_url(), AzureAuthMode::Bearer)
            .embed_batch("tok", "my-deployment", &["a".to_string(), "b".to_string()])
            .await;

        assert!(result.is_err(), "a bogus index must be reported, never used to index out of bounds");
    }

    #[test]
    fn cache_scope_separates_resources_surfaces_and_input_types() {
        let a = new_provider(
            "https://res-a.openai.azure.com",
            Some("https://res-a.services.ai.azure.com".to_string()),
            "2024-05-01-preview",
            AzureAuthMode::Bearer,
        );
        let b = new_provider(
            "https://res-b.openai.azure.com",
            Some("https://res-b.services.ai.azure.com".to_string()),
            "2024-05-01-preview",
            AzureAuthMode::Bearer,
        );

        assert_ne!(
            a.cache_scope("shared-deployment-name").unwrap(),
            b.cache_scope("shared-deployment-name").unwrap(),
            "two Azure resources can name two different models identically"
        );
        assert_ne!(
            a.cache_scope("shared-name").unwrap(),
            a.cache_scope("foundry:shared-name").unwrap()
        );
        assert_ne!(
            a.cache_scope("foundry:m#document").unwrap(),
            a.cache_scope("foundry:m#query").unwrap()
        );
    }

    #[test]
    fn cache_scope_rejects_a_malformed_model_rather_than_inventing_a_scope() {
        let p = new_provider("https://res.openai.azure.com", None, "2024-05-01-preview", AzureAuthMode::Bearer);
        assert!(p.cache_scope("foundry:cohere-embed-v3-english").is_err());
        assert!(p.cache_scope("").is_err());
    }
}

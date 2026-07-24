use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::{Deserialize, Serialize};
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

// Deliberately conservative and uniform across all three provider adapters
// rather than tuned to each provider's real limit — that real limit could
// not be reliably verified (Mistral only documents a token limit, not an
// item count; Gemini's known "150" figure belongs to a different, async
// batch-job product, not this synchronous endpoint). Staying well under any
// plausible limit avoids the whole verification problem.
const MAX_BATCH_SIZE: usize = 100;

// A hung upstream connection must not block a request indefinitely -- 30s
// is a conservative ceiling for a same-region HTTPS call to a major
// provider's embeddings endpoint, not a measured SLA (none of the three
// providers publish one). Uniform across adapters for the same reason
// MAX_BATCH_SIZE is uniform: no verified per-provider number to tune to.
const PROVIDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

// A transient upstream failure (429 rate limit, 5xx, or a connection
// blip -- exactly what a real 429 from Gemini surfaced during battle-
// testing on 2026-07-24) shouldn't fail the whole batch on the first
// try. reqwest-retry's DefaultRetryableStrategy already does the right
// thing with no configuration: retries on 5xx / 408 / 429 / connection
// errors, never on other 4xx (400/401/404/422 are never going to
// succeed on retry, so failing fast on those is correct, not a gap).
// 3 retries, exponential backoff with jitter -- conservative and
// uniform across adapters for the same reason MAX_BATCH_SIZE and
// PROVIDER_TIMEOUT are: no verified per-provider number to tune to.
const MAX_RETRIES: u32 = 3;

pub struct OpenAiProvider {
    client: reqwest_middleware::ClientWithMiddleware,
    base_url: String,
}

impl OpenAiProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(PROVIDER_TIMEOUT)
            .build()
            .expect("reqwest client with a timeout is always constructible");
        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(MAX_RETRIES);
        Self {
            client: ClientBuilder::new(inner)
                .with(RetryTransientMiddleware::new_with_policy(retry_policy))
                .build(),
            base_url: base_url.into(),
        }
    }
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
    usage: UsageResponse,
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

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAiProvider {
    async fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let mut ordered = vec![Vec::new(); texts.len()];
        let mut prompt_tokens = 0;
        let mut total_tokens = 0;

        for (chunk_index, chunk) in texts.chunks(MAX_BATCH_SIZE).enumerate() {
            let base_index = chunk_index * MAX_BATCH_SIZE;
            let body = EmbeddingsRequest { model, input: chunk };

            let response = self
                .client
                .post(format!("{}/v1/embeddings", self.base_url))
                .bearer_auth(api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| ProviderError(e.to_string()))?
                .error_for_status()
                .map_err(|e| ProviderError(e.to_string()))?
                .json::<EmbeddingsResponse>()
                .await
                .map_err(|e| ProviderError(e.to_string()))?;

            for item in response.data {
                ordered[base_index + item.index] = item.embedding;
            }
            prompt_tokens += response.usage.prompt_tokens;
            total_tokens += response.usage.total_tokens;
        }

        let usage = ProviderUsage { prompt_tokens, total_tokens };
        Ok((ordered, usage))
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn embed_batch_reorders_response_by_index_and_returns_usage() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/embeddings")
                    .header("authorization", "Bearer test-key")
                    .json_body(json!({ "model": "text-embedding-3-small", "input": ["a", "b"] }));
                then.status(200).json_body(json!({
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        { "embedding": [2.0], "index": 1 },
                        { "embedding": [1.0], "index": 0 }
                    ],
                    "usage": { "prompt_tokens": 5, "total_tokens": 5 }
                }));
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let (vectors, usage) = provider
            .embed_batch("test-key", "text-embedding-3-small", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_http_error_status() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/embeddings");
                then.status(401).json_body(json!({ "error": "invalid api key" }));
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider.embed_batch("bad-key", "m", &["x".to_string()]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_malformed_response_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/embeddings");
                then.status(200).body("not json");
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider.embed_batch("test-key", "m", &["x".to_string()]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_splits_large_input_into_chunks_and_reassembles_in_order() {
        let server = MockServer::start_async().await;

        // 150 inputs with MAX_BATCH_SIZE=100 must produce exactly two calls:
        // one with 100 items, one with 50.
        let first_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/embeddings")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                        body["input"].as_array().map(|a| a.len()) == Some(100)
                    });
                then.status(200).json_body_obj(&json!({
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": (0..100).map(|i| json!({ "embedding": [i as f64], "index": i })).collect::<Vec<_>>(),
                    "usage": { "prompt_tokens": 100, "total_tokens": 100 }
                }));
            })
            .await;
        let second_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/embeddings")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                        body["input"].as_array().map(|a| a.len()) == Some(50)
                    });
                then.status(200).json_body_obj(&json!({
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": (0..50).map(|i| json!({ "embedding": [1000.0 + i as f64], "index": i })).collect::<Vec<_>>(),
                    "usage": { "prompt_tokens": 50, "total_tokens": 50 }
                }));
            })
            .await;

        let texts: Vec<String> = (0..150).map(|i| format!("text-{i}")).collect();
        let provider = OpenAiProvider::new(server.base_url());
        let (vectors, usage) = provider
            .embed_batch("test-key", "text-embedding-3-small", &texts)
            .await
            .unwrap();

        first_chunk.assert_async().await;
        second_chunk.assert_async().await;
        assert_eq!(vectors.len(), 150);
        assert_eq!(vectors[0], vec![0.0]);
        assert_eq!(vectors[99], vec![99.0]);
        assert_eq!(vectors[100], vec![1000.0]);
        assert_eq!(vectors[149], vec![1049.0]);
        assert_eq!(usage.prompt_tokens, 150);
        assert_eq!(usage.total_tokens, 150);
    }

    #[tokio::test]
    async fn embed_batch_retries_on_transient_5xx_before_giving_up() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/embeddings");
                then.status(503).body("service unavailable");
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider.embed_batch("test-key", "m", &["x".to_string()]).await;

        assert!(result.is_err(), "must still fail once retries are exhausted, not hang or silently succeed");
        assert_eq!(
            mock.hits_async().await,
            (MAX_RETRIES + 1) as usize,
            "1 initial attempt + MAX_RETRIES retries -- proves retry actually happened, not just that failure was reported"
        );
    }

    #[tokio::test]
    async fn embed_batch_does_not_retry_on_a_fatal_4xx_error() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/embeddings");
                then.status(401).json_body(json!({ "error": "invalid api key" }));
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider.embed_batch("bad-key", "m", &["x".to_string()]).await;

        assert!(result.is_err());
        assert_eq!(
            mock.hits_async().await,
            1,
            "a 401 will never succeed on retry -- retrying it would only slow down a real auth failure for no benefit"
        );
    }
}

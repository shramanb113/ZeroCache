use serde::{Deserialize, Serialize};
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

// See zerocache-adapters-openai for why this is a single conservative
// constant shared uniformly across adapters rather than a tuned per-provider
// limit.
const MAX_BATCH_SIZE: usize = 100;

pub struct GeminiProvider {
    client: reqwest::Client,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
        }
    }
}

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<EmbedContentRequest>,
}

#[derive(Serialize)]
struct EmbedContentRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<ContentEmbedding>,
}

#[derive(Deserialize)]
struct ContentEmbedding {
    values: Vec<f32>,
}

#[async_trait::async_trait]
impl EmbeddingProvider for GeminiProvider {
    async fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let qualified_model = format!("models/{model}");
        let mut vectors = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let body = BatchEmbedRequest {
                requests: chunk
                    .iter()
                    .map(|text| EmbedContentRequest {
                        model: qualified_model.clone(),
                        content: Content { parts: vec![Part { text: text.clone() }] },
                    })
                    .collect(),
            };

            let response = self
                .client
                .post(format!("{}/v1beta/models/{model}:batchEmbedContents", self.base_url))
                .header("x-goog-api-key", api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| ProviderError(e.to_string()))?
                .error_for_status()
                .map_err(|e| ProviderError(e.to_string()))?
                .json::<BatchEmbedResponse>()
                .await
                .map_err(|e| ProviderError(e.to_string()))?;

            if response.embeddings.len() != chunk.len() {
                return Err(ProviderError(format!(
                    "expected {} embeddings in response, got {}",
                    chunk.len(),
                    response.embeddings.len()
                )));
            }

            vectors.extend(response.embeddings.into_iter().map(|e| e.values));
        }

        // Gemini's embedding API does not report token usage.
        Ok((vectors, ProviderUsage::default()))
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
    async fn embed_batch_sends_x_goog_api_key_not_bearer_auth() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1beta/models/text-embedding-004:batchEmbedContents")
                    .header("x-goog-api-key", "test-key")
                    .json_body(json!({
                        "requests": [
                            { "model": "models/text-embedding-004", "content": { "parts": [{ "text": "a" }] } },
                            { "model": "models/text-embedding-004", "content": { "parts": [{ "text": "b" }] } }
                        ]
                    }));
                then.status(200).json_body(json!({
                    "embeddings": [
                        { "values": [1.0] },
                        { "values": [2.0] }
                    ]
                }));
            })
            .await;

        let provider = GeminiProvider::new(server.base_url());
        let (vectors, usage) = provider
            .embed_batch("test-key", "text-embedding-004", &["a".to_string(), "b".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(usage.prompt_tokens, 0, "Gemini does not report usage; must stay zero, not fabricated");
        assert_eq!(usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_http_error_status() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
                then.status(403).json_body(json!({ "error": { "message": "API key not valid" } }));
            })
            .await;

        let provider = GeminiProvider::new(server.base_url());
        let result = provider.embed_batch("bad-key", "text-embedding-004", &["x".to_string()]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_malformed_response_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
                then.status(200).body("not json");
            })
            .await;

        let provider = GeminiProvider::new(server.base_url());
        let result = provider.embed_batch("test-key", "text-embedding-004", &["x".to_string()]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_returns_error_when_embedding_count_does_not_match_request_count() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
                then.status(200).json_body(json!({ "embeddings": [{ "values": [1.0] }] }));
            })
            .await;

        let provider = GeminiProvider::new(server.base_url());
        let result = provider
            .embed_batch("test-key", "text-embedding-004", &["a".to_string(), "b".to_string()])
            .await;

        assert!(result.is_err(), "a count mismatch must be a hard error, not a silent misalignment");
    }

    #[tokio::test]
    async fn embed_batch_splits_large_input_into_chunks_and_concatenates_in_order() {
        let server = MockServer::start_async().await;

        let first_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1beta/models/text-embedding-004:batchEmbedContents")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                        body["requests"].as_array().map(|a| a.len()) == Some(100)
                    });
                then.status(200).json_body_obj(&json!({
                    "embeddings": (0..100).map(|i| json!({ "values": [i as f64] })).collect::<Vec<_>>()
                }));
            })
            .await;
        let second_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1beta/models/text-embedding-004:batchEmbedContents")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default()).unwrap();
                        body["requests"].as_array().map(|a| a.len()) == Some(50)
                    });
                then.status(200).json_body_obj(&json!({
                    "embeddings": (0..50).map(|i| json!({ "values": [1000.0 + i as f64] })).collect::<Vec<_>>()
                }));
            })
            .await;

        let texts: Vec<String> = (0..150).map(|i| format!("text-{i}")).collect();
        let provider = GeminiProvider::new(server.base_url());
        let (vectors, _usage) = provider
            .embed_batch("test-key", "text-embedding-004", &texts)
            .await
            .unwrap();

        first_chunk.assert_async().await;
        second_chunk.assert_async().await;
        assert_eq!(vectors.len(), 150);
        assert_eq!(vectors[0], vec![0.0]);
        assert_eq!(vectors[99], vec![99.0]);
        assert_eq!(vectors[100], vec![1000.0]);
        assert_eq!(vectors[149], vec![1049.0]);
    }
}

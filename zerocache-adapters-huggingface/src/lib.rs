use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::Serialize;
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

// HuggingFace documents no per-call batch-size limit for `inputs` -- same
// unverified-conservative posture as the other three adapters' MAX_BATCH_SIZE.
const MAX_BATCH_SIZE: usize = 100;

const PROVIDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

const MAX_RETRIES: u32 = 3;

pub struct HuggingFaceProvider {
    client: reqwest_middleware::ClientWithMiddleware,
    base_url: String,
}

impl HuggingFaceProvider {
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
struct FeatureExtractionRequest<'a> {
    inputs: &'a [String],
}

#[async_trait::async_trait]
impl EmbeddingProvider for HuggingFaceProvider {
    async fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let mut vectors = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let body = FeatureExtractionRequest { inputs: chunk };

            let response: Vec<Vec<f32>> = self
                .client
                .post(format!(
                    "{}/models/{model}/pipeline/feature-extraction",
                    self.base_url
                ))
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&body)
                .send()
                .await
                .map_err(|e| ProviderError(e.to_string()))?
                .error_for_status()
                .map_err(|e| ProviderError(e.to_string()))?
                .json()
                .await
                .map_err(|e| ProviderError(e.to_string()))?;

            if response.len() != chunk.len() {
                return Err(ProviderError(format!(
                    "expected {} embeddings in response, got {}",
                    chunk.len(),
                    response.len()
                )));
            }

            vectors.extend(response);
        }

        // HuggingFace's feature-extraction response has no usage/token field
        // at all -- never reported, same posture as Gemini.
        Ok((vectors, ProviderUsage::default()))
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
        // This adapter's wire shape is fixed, so the only thing that can vary
        // between two instances is which endpoint they talk to.
        Ok(self.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn embed_batch_sends_bearer_auth_and_model_in_the_url_path() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction")
                    .header("Authorization", "Bearer test-key")
                    .json_body(json!({ "inputs": ["a", "b"] }));
                then.status(200).json_body(json!([[1.0, 2.0], [3.0, 4.0]]));
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let (vectors, usage) = provider
            .embed_batch(
                "test-key",
                "sentence-transformers/all-MiniLM-L6-v2",
                &["a".to_string(), "b".to_string()],
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(usage.prompt_tokens, 0, "HuggingFace's feature-extraction response has no usage field at all -- must stay zero, not fabricated");
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_http_error_status() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction");
                then.status(401)
                    .json_body(json!({ "error": "Invalid token" }));
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let result = provider
            .embed_batch("bad-key", "some-model", &["x".to_string()])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_returns_error_on_malformed_response_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction");
                then.status(200).body("not json");
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let result = provider
            .embed_batch("test-key", "some-model", &["x".to_string()])
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn embed_batch_returns_error_when_vector_count_does_not_match_text_count() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction");
                then.status(200).json_body(json!([[1.0]]));
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let result = provider
            .embed_batch(
                "test-key",
                "some-model",
                &["a".to_string(), "b".to_string()],
            )
            .await;

        assert!(
            result.is_err(),
            "a count mismatch must be a hard error, same discipline as the other three adapters"
        );
    }

    #[tokio::test]
    async fn embed_batch_splits_large_input_into_chunks_and_concatenates_in_order() {
        let server = MockServer::start_async().await;

        let first_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default())
                                .unwrap();
                        body["inputs"].as_array().map(|a| a.len()) == Some(100)
                    });
                then.status(200)
                    .json_body_obj(&json!((0..100).map(|i| vec![i as f64]).collect::<Vec<_>>()));
            })
            .await;
        let second_chunk = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction")
                    .matches(|req| {
                        let body: serde_json::Value =
                            serde_json::from_slice(req.body.as_deref().unwrap_or_default())
                                .unwrap();
                        body["inputs"].as_array().map(|a| a.len()) == Some(50)
                    });
                then.status(200).json_body_obj(&json!((0..50)
                    .map(|i| vec![1000.0 + i as f64])
                    .collect::<Vec<_>>()));
            })
            .await;

        let texts: Vec<String> = (0..150).map(|i| format!("text-{i}")).collect();
        let provider = HuggingFaceProvider::new(server.base_url());
        let (vectors, _usage) = provider
            .embed_batch("test-key", "some-model", &texts)
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

    #[tokio::test]
    async fn embed_batch_retries_on_transient_5xx_before_giving_up() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction");
                then.status(503)
                    .json_body(json!({ "error": "Model is overloaded" }));
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let result = provider
            .embed_batch("test-key", "some-model", &["x".to_string()])
            .await;

        assert!(
            result.is_err(),
            "must still fail once retries are exhausted, not hang or silently succeed"
        );
        assert_eq!(mock.hits_async().await, (MAX_RETRIES + 1) as usize);
    }

    #[tokio::test]
    async fn embed_batch_does_not_retry_on_a_fatal_4xx_error() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/some-model/pipeline/feature-extraction");
                then.status(401)
                    .json_body(json!({ "error": "Invalid token" }));
            })
            .await;

        let provider = HuggingFaceProvider::new(server.base_url());
        let result = provider
            .embed_batch("bad-key", "some-model", &["x".to_string()])
            .await;

        assert!(result.is_err());
        assert_eq!(
            mock.hits_async().await,
            1,
            "a 401 will never succeed on retry"
        );
    }
}

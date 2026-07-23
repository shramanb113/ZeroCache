use serde::{Deserialize, Serialize};
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

pub struct GeminiProvider {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
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

impl EmbeddingProvider for GeminiProvider {
    fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let qualified_model = format!("models/{model}");
        let body = BatchEmbedRequest {
            requests: texts
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
            .map_err(|e| ProviderError(e.to_string()))?
            .error_for_status()
            .map_err(|e| ProviderError(e.to_string()))?
            .json::<BatchEmbedResponse>()
            .map_err(|e| ProviderError(e.to_string()))?;

        if response.embeddings.len() != texts.len() {
            return Err(ProviderError(format!(
                "expected {} embeddings in response, got {}",
                texts.len(),
                response.embeddings.len()
            )));
        }

        let vectors = response.embeddings.into_iter().map(|e| e.values).collect();

        // Gemini's embedding API does not report token usage.
        Ok((vectors, ProviderUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    #[test]
    fn embed_batch_sends_x_goog_api_key_not_bearer_auth() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
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
        });

        let provider = GeminiProvider::new(server.base_url());
        let (vectors, usage) = provider
            .embed_batch("test-key", "text-embedding-004", &["a".to_string(), "b".to_string()])
            .unwrap();

        mock.assert();
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(usage.prompt_tokens, 0, "Gemini does not report usage; must stay zero, not fabricated");
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn embed_batch_returns_error_on_http_error_status() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
            then.status(403).json_body(json!({ "error": { "message": "API key not valid" } }));
        });

        let provider = GeminiProvider::new(server.base_url());
        let result = provider.embed_batch("bad-key", "text-embedding-004", &["x".to_string()]);

        assert!(result.is_err());
    }

    #[test]
    fn embed_batch_returns_error_on_malformed_response_body() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
            then.status(200).body("not json");
        });

        let provider = GeminiProvider::new(server.base_url());
        let result = provider.embed_batch("test-key", "text-embedding-004", &["x".to_string()]);

        assert!(result.is_err());
    }

    #[test]
    fn embed_batch_returns_error_when_embedding_count_does_not_match_request_count() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1beta/models/text-embedding-004:batchEmbedContents");
            // two texts requested, only one embedding returned
            then.status(200).json_body(json!({ "embeddings": [{ "values": [1.0] }] }));
        });

        let provider = GeminiProvider::new(server.base_url());
        let result = provider.embed_batch(
            "test-key",
            "text-embedding-004",
            &["a".to_string(), "b".to_string()],
        );

        assert!(result.is_err(), "a count mismatch must be a hard error, not a silent misalignment");
    }
}

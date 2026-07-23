use serde::{Deserialize, Serialize};
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

pub struct OpenAiProvider {
    client: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
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

impl EmbeddingProvider for OpenAiProvider {
    fn embed_batch(
        &self,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let body = EmbeddingsRequest { model, input: texts };

        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .map_err(|e| ProviderError(e.to_string()))?
            .error_for_status()
            .map_err(|e| ProviderError(e.to_string()))?
            .json::<EmbeddingsResponse>()
            .map_err(|e| ProviderError(e.to_string()))?;

        let mut ordered = vec![Vec::new(); texts.len()];
        for item in response.data {
            ordered[item.index] = item.embedding;
        }

        let usage = ProviderUsage {
            prompt_tokens: response.usage.prompt_tokens,
            total_tokens: response.usage.total_tokens,
        };

        Ok((ordered, usage))
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    #[test]
    fn embed_batch_reorders_response_by_index_and_returns_usage() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .header("authorization", "Bearer test-key")
                .json_body(json!({ "model": "text-embedding-3-small", "input": ["a", "b"] }));
            then.status(200).json_body(json!({
                "object": "list",
                "model": "text-embedding-3-small",
                // returned out of order on purpose, to prove the adapter
                // reorders by `index` rather than trusting array order
                "data": [
                    { "embedding": [2.0], "index": 1 },
                    { "embedding": [1.0], "index": 0 }
                ],
                "usage": { "prompt_tokens": 5, "total_tokens": 5 }
            }));
        });

        let provider = OpenAiProvider::new(server.base_url(), "test-key");
        let (vectors, usage) = provider
            .embed_batch("text-embedding-3-small", &["a".to_string(), "b".to_string()])
            .unwrap();

        mock.assert();
        assert_eq!(vectors, vec![vec![1.0], vec![2.0]]);
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.total_tokens, 5);
    }

    #[test]
    fn embed_batch_returns_error_on_http_error_status() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(401).json_body(json!({ "error": "invalid api key" }));
        });

        let provider = OpenAiProvider::new(server.base_url(), "bad-key");
        let result = provider.embed_batch("m", &["x".to_string()]);

        assert!(result.is_err());
    }

    #[test]
    fn embed_batch_returns_error_on_malformed_response_body() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(200).body("not json");
        });

        let provider = OpenAiProvider::new(server.base_url(), "test-key");
        let result = provider.embed_batch("m", &["x".to_string()]);

        assert!(result.is_err());
    }
}

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

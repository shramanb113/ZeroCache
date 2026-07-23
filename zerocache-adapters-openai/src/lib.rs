use serde::{Deserialize, Serialize};
use zerocache_ports::EmbeddingProvider;

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
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

impl EmbeddingProvider for OpenAiProvider {
    fn embed_batch(&self, model: &str, texts: &[String]) -> Vec<Vec<f32>> {
        let body = EmbeddingsRequest { model, input: texts };

        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .expect("embedding provider request failed")
            .error_for_status()
            .expect("embedding provider returned an error status")
            .json::<EmbeddingsResponse>()
            .expect("embedding provider response did not match the expected shape");

        let mut ordered = vec![Vec::new(); texts.len()];
        for item in response.data {
            ordered[item.index] = item.embedding;
        }
        ordered
    }
}

use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::{Deserialize, Serialize};
use zerocache_ports::{
    ChatCompletionProvider, ChatCompletionResponse, CompletionUsage, EmbeddingProvider,
    ProviderError, ProviderUsage,
};

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
            let body = EmbeddingsRequest {
                model,
                input: chunk,
            };

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

        let usage = ProviderUsage {
            prompt_tokens,
            total_tokens,
        };
        Ok((ordered, usage))
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

/// A generic OpenAI-wire chat-completions proxy. Takes an explicit URL
/// prefix and appends only `/chat/completions`, so it serves any endpoint
/// speaking OpenAI's chat wire shape under any path (OpenAI, Gemini's
/// `/v1beta/openai` compat surface, Groq, self-hosted vLLM). Distinct from
/// `OpenAiProvider`, which is the embeddings adapter and hardcodes `/v1/`.
pub struct OpenAiWireChatProvider {
    client: reqwest_middleware::ClientWithMiddleware,
    url_prefix: String,
}

impl OpenAiWireChatProvider {
    /// `url_prefix` is the endpoint up to but not including
    /// `/chat/completions`, already run through `normalize_chat_url`.
    pub fn new(url_prefix: impl Into<String>) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(PROVIDER_TIMEOUT)
            .build()
            .expect("reqwest client with a timeout is always constructible");
        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(MAX_RETRIES);
        Self {
            client: ClientBuilder::new(inner)
                .with(RetryTransientMiddleware::new_with_policy(retry_policy))
                .build(),
            url_prefix: url_prefix.into(),
        }
    }
}

#[async_trait::async_trait]
impl ChatCompletionProvider for OpenAiWireChatProvider {
    async fn chat_completion(
        &self,
        api_key: &str,
        request: &serde_json::Value,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        // Forward the body verbatim, swap in the caller's key. `Err` means a
        // transport failure; a non-2xx response is returned as `Ok` with its
        // real status so the caller sees exactly what the provider said.
        let response = self
            .client
            .post(format!("{}/chat/completions", self.url_prefix))
            .bearer_auth(api_key)
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;

        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;
        // A non-2xx body may not be JSON (a gateway's plain-text 503); wrap
        // it rather than fail a forwardable error response.
        let body: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({ "error": text }));

        let usage = CompletionUsage {
            prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        };

        Ok(ChatCompletionResponse {
            status,
            body,
            usage,
        })
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
        Ok(self.url_prefix.clone())
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
            .embed_batch(
                "test-key",
                "text-embedding-3-small",
                &["a".to_string(), "b".to_string()],
            )
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
                then.status(401)
                    .json_body(json!({ "error": "invalid api key" }));
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider
            .embed_batch("bad-key", "m", &["x".to_string()])
            .await;

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
        let result = provider
            .embed_batch("test-key", "m", &["x".to_string()])
            .await;

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
        let result = provider
            .embed_batch("test-key", "m", &["x".to_string()])
            .await;

        assert!(
            result.is_err(),
            "must still fail once retries are exhausted, not hang or silently succeed"
        );
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
                then.status(401)
                    .json_body(json!({ "error": "invalid api key" }));
            })
            .await;

        let provider = OpenAiProvider::new(server.base_url());
        let result = provider
            .embed_batch("bad-key", "m", &["x".to_string()])
            .await;

        assert!(result.is_err());
        assert_eq!(
            mock.hits_async().await,
            1,
            "a 401 will never succeed on retry -- retrying it would only slow down a real auth failure for no benefit"
        );
    }

    #[test]
    fn cache_scope_is_the_configured_base_url_so_repointing_invalidates_the_cache() {
        let a = OpenAiProvider::new("https://api.openai.com");
        let b = OpenAiProvider::new("http://localhost:8000");
        assert_eq!(
            a.cache_scope("text-embedding-3-small").unwrap(),
            "https://api.openai.com"
        );
        assert_ne!(
            a.cache_scope("text-embedding-3-small").unwrap(),
            b.cache_scope("text-embedding-3-small").unwrap(),
            "a self-hosted endpoint must not inherit vectors cached from the real provider"
        );
    }

    #[tokio::test]
    async fn chat_completion_forwards_the_body_and_returns_the_response_with_usage() {
        let server = MockServer::start_async().await;
        let request_body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0
        });
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/chat/completions")
                    .header("authorization", "Bearer sk-caller")
                    .json_body(json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                        "temperature": 0
                    }));
                then.status(200).json_body(json!({
                    "id": "chatcmpl-1",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
                }));
            })
            .await;

        let provider = OpenAiWireChatProvider::new(server.base_url());
        let resp = provider
            .chat_completion("sk-caller", &request_body)
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body["choices"][0]["message"]["content"], "hello");
        assert_eq!(resp.usage.prompt_tokens, 9);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn chat_completion_surfaces_a_non_2xx_as_ok_with_the_real_status_and_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/chat/completions");
                then.status(429).json_body(
                    json!({"error": {"message": "rate limited", "type": "rate_limit_error"}}),
                );
            })
            .await;

        let provider = OpenAiWireChatProvider::new(server.base_url());
        let resp = provider
            .chat_completion("sk-caller", &json!({"model": "gpt-4o", "messages": []}))
            .await
            .expect("a non-2xx upstream response is not a transport error");

        assert_eq!(resp.status, 429);
        assert_eq!(resp.body["error"]["type"], "rate_limit_error");
        assert_eq!(resp.usage.prompt_tokens, 0);
    }

    #[tokio::test]
    async fn chat_completion_defaults_usage_to_zero_when_the_response_has_no_usage_block() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/chat/completions");
                then.status(200).json_body(json!({
                    "id": "chatcmpl-2",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}]
                }));
            })
            .await;

        let provider = OpenAiWireChatProvider::new(server.base_url());
        let resp = provider
            .chat_completion("sk-caller", &json!({"model": "gpt-4o", "messages": []}))
            .await
            .unwrap();

        assert_eq!(resp.status, 200);
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert_eq!(resp.usage.completion_tokens, 0);
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn chat_completion_retries_a_transient_5xx_then_forwards_the_final_status() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/chat/completions");
                then.status(503).body("service unavailable");
            })
            .await;

        let provider = OpenAiWireChatProvider::new(server.base_url());
        let resp = provider
            .chat_completion("sk-caller", &json!({"model": "gpt-4o", "messages": []}))
            .await
            .unwrap();

        assert_eq!(
            resp.status, 503,
            "the final upstream status is forwarded, not turned into an Err"
        );
        assert_eq!(
            mock.hits_async().await,
            (MAX_RETRIES + 1) as usize,
            "1 initial attempt + MAX_RETRIES retries"
        );
    }

    #[tokio::test]
    async fn chat_completion_appends_only_chat_completions_to_a_prefixed_url() {
        // A prefix that already carries a version segment (/v1beta/openai)
        // must not get a second /v1.
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1beta/openai/chat/completions");
                then.status(200).json_body(json!({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }));
            })
            .await;

        let provider = OpenAiWireChatProvider::new(format!("{}/v1beta/openai", server.base_url()));
        let resp = provider
            .chat_completion(
                "sk-caller",
                &json!({"model": "gemini-3.5-flash-lite", "messages": []}),
            )
            .await
            .unwrap();

        mock.assert_async().await; // fails if the request hit a doubled-/v1 path
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn chat_completion_cache_scope_is_the_configured_url_prefix() {
        let a = OpenAiWireChatProvider::new("https://api.openai.com/v1");
        let b =
            OpenAiWireChatProvider::new("https://generativelanguage.googleapis.com/v1beta/openai");
        assert_eq!(
            a.cache_scope("gpt-4o").unwrap(),
            "https://api.openai.com/v1"
        );
        assert_ne!(
            a.cache_scope("gpt-4o").unwrap(),
            b.cache_scope("gpt-4o").unwrap(),
            "real OpenAI and a Gemini-compat endpoint must not share completion-cache entries"
        );
    }
}

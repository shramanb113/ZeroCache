//! Anthropic native `/v1/messages` proxy adapter. Mirrors the chat half of
//! `zerocache-adapters-openai::OpenAiWireChatProvider`: forwards the request
//! body verbatim, swaps in the caller's key as `x-api-key`, and returns a
//! non-2xx as `Ok` with its real status so the proxy forwards error bodies
//! unchanged and never caches them.

use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use zerocache_ports::{
    ChatCompletionResponse, CompletionUsage, MessageHeaders, MessagesProvider, ProviderError,
    SseByteStream,
};

/// Same conservative, unmeasured ceiling as every other provider adapter.
const PROVIDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Same retry budget as every other provider adapter (5xx / 408 / 429 /
/// connection errors only, via `reqwest-retry`'s default strategy).
const MAX_RETRIES: u32 = 3;
/// Sent when the caller supplied no `anthropic-version` header. Anthropic
/// treats the header as optional but recommends always sending one.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicMessagesProvider {
    client: reqwest_middleware::ClientWithMiddleware,
    /// Streaming keeps `PROVIDER_TIMEOUT` only as a per-read idle cap (a long
    /// generation is the point of `stream: true`) and has no retry middleware
    /// — a partially consumed stream must not be replayed. Same shape as
    /// `OpenAiWireChatProvider::stream_client`.
    stream_client: reqwest::Client,
    /// Bare origin, e.g. `https://api.anthropic.com`; the adapter appends
    /// `/v1/messages`.
    base_url: String,
}

impl AnthropicMessagesProvider {
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
            stream_client: reqwest::Client::builder()
                .read_timeout(PROVIDER_TIMEOUT)
                .build()
                .expect("reqwest client with a read timeout is always constructible"),
            base_url: base_url.into(),
        }
    }

    fn url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }
}

/// One `header(k, v) -> Self` call, over either builder type. The buffered
/// path yields a `reqwest_middleware::RequestBuilder` and the streaming path a
/// plain `reqwest::RequestBuilder`; the two share no trait, so `apply_headers`
/// is generic over this.
trait HeaderSink: Sized {
    fn with_header(self, key: &str, value: &str) -> Self;
}

impl HeaderSink for reqwest::RequestBuilder {
    fn with_header(self, key: &str, value: &str) -> Self {
        self.header(key, value)
    }
}

impl HeaderSink for reqwest_middleware::RequestBuilder {
    fn with_header(self, key: &str, value: &str) -> Self {
        self.header(key, value)
    }
}

/// Applies the Anthropic auth + version + optional beta headers to a request
/// builder. Shared by both the buffered and streaming paths.
fn apply_headers<B: HeaderSink>(builder: B, api_key: &str, headers: &MessageHeaders) -> B {
    let version = headers
        .anthropic_version
        .as_deref()
        .unwrap_or(DEFAULT_ANTHROPIC_VERSION);
    let mut b = builder
        .with_header("x-api-key", api_key)
        .with_header("anthropic-version", version);
    if let Some(beta) = &headers.anthropic_beta {
        b = b.with_header("anthropic-beta", beta);
    }
    b
}

#[async_trait::async_trait]
impl MessagesProvider for AnthropicMessagesProvider {
    async fn messages(
        &self,
        api_key: &str,
        request: &serde_json::Value,
        headers: &MessageHeaders,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let response = apply_headers(self.client.post(self.url()), api_key, headers)
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;

        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;
        // A non-2xx body may not be JSON (a gateway's plain-text 503); wrap it
        // rather than fail a forwardable error response.
        let body: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({ "error": text }));

        let input = body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
        let output = body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
        let usage = CompletionUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
        };

        Ok(ChatCompletionResponse {
            status,
            body,
            usage,
        })
    }

    async fn messages_stream_passthrough(
        &self,
        api_key: &str,
        request: &serde_json::Value,
        headers: &MessageHeaders,
    ) -> Result<(u16, SseByteStream), ProviderError> {
        let response = apply_headers(self.stream_client.post(self.url()), api_key, headers)
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError(e.to_string()))?;

        let status = response.status().as_u16();
        let stream = futures::StreamExt::map(response.bytes_stream(), |chunk| {
            chunk
                .map(|b| b.to_vec())
                .map_err(|e| ProviderError(e.to_string()))
        });
        Ok((status, Box::pin(stream)))
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
        // Fixed wire shape, so the only thing that varies between instances is
        // which endpoint they talk to (repointing => cold cache).
        Ok(self.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    fn ok_message_body() -> serde_json::Value {
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 3}
        })
    }

    #[tokio::test]
    async fn messages_forwards_the_body_and_maps_usage() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/messages")
                    .header("x-api-key", "sk-caller")
                    .header("anthropic-version", DEFAULT_ANTHROPIC_VERSION)
                    .json_body(json!({
                        "model": "claude-opus-4-6",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8,
                        "temperature": 0
                    }));
                then.status(200).json_body(ok_message_body());
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        let resp = provider
            .messages(
                "sk-caller",
                &json!({
                    "model": "claude-opus-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                    "temperature": 0
                }),
                &MessageHeaders::default(),
            )
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body["content"][0]["text"], "hello");
        assert_eq!(resp.usage.prompt_tokens, 9);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn messages_uses_x_api_key_not_bearer_auth() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/messages")
                    .header_exists("x-api-key")
                    .matches(|req| {
                        !req.headers
                            .as_ref()
                            .map(|h| {
                                h.iter()
                                    .any(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                            })
                            .unwrap_or(false)
                    });
                then.status(200).json_body(ok_message_body());
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        provider
            .messages(
                "sk-caller",
                &json!({"messages": []}),
                &MessageHeaders::default(),
            )
            .await
            .unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn messages_forwards_a_caller_supplied_version_and_beta_header() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/messages")
                    .header("anthropic-version", "2026-01-01")
                    .header("anthropic-beta", "output-128k-2025-02-19");
                then.status(200).json_body(ok_message_body());
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        provider
            .messages(
                "sk-caller",
                &json!({"messages": []}),
                &MessageHeaders {
                    anthropic_version: Some("2026-01-01".into()),
                    anthropic_beta: Some("output-128k-2025-02-19".into()),
                },
            )
            .await
            .unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn messages_surfaces_a_non_2xx_as_ok_with_the_real_status_and_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/messages");
                then.status(400).json_body(json!({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "temperature: 0 is not supported"}
                }));
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        let resp = provider
            .messages(
                "sk-caller",
                &json!({"messages": []}),
                &MessageHeaders::default(),
            )
            .await
            .expect("a non-2xx upstream response is not a transport error");

        assert_eq!(resp.status, 400);
        assert_eq!(resp.body["error"]["type"], "invalid_request_error");
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn messages_wraps_a_non_json_error_body() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/messages");
                then.status(503).body("upstream down");
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        let resp = provider
            .messages(
                "sk-caller",
                &json!({"messages": []}),
                &MessageHeaders::default(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status, 503);
        assert_eq!(resp.body["error"], "upstream down");
    }

    #[tokio::test]
    async fn messages_retries_a_transient_5xx_then_forwards_the_final_status() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/messages");
                then.status(503).body("service unavailable");
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        let resp = provider
            .messages(
                "sk-caller",
                &json!({"messages": []}),
                &MessageHeaders::default(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status, 503);
        assert_eq!(mock.hits_async().await, (MAX_RETRIES + 1) as usize);
    }

    #[test]
    fn cache_scope_is_the_configured_base_url() {
        let a = AnthropicMessagesProvider::new("https://api.anthropic.com");
        let b = AnthropicMessagesProvider::new("https://gw.internal/anthropic");
        assert_eq!(
            a.cache_scope("claude-opus-4-6").unwrap(),
            "https://api.anthropic.com"
        );
        assert_ne!(
            a.cache_scope("claude-opus-4-6").unwrap(),
            b.cache_scope("claude-opus-4-6").unwrap()
        );
    }

    #[tokio::test]
    async fn stream_passthrough_yields_upstream_bytes_in_order_and_surfaces_a_non_2xx() {
        let server = MockServer::start_async().await;
        let body = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n\
                    event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/messages");
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body(body);
            })
            .await;

        let provider = AnthropicMessagesProvider::new(server.base_url());
        let (status, mut stream) = provider
            .messages_stream_passthrough(
                "sk-caller",
                &json!({"messages": [], "stream": true}),
                &MessageHeaders::default(),
            )
            .await
            .unwrap();

        assert_eq!(status, 200);
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(String::from_utf8(collected).unwrap(), body);
    }
}

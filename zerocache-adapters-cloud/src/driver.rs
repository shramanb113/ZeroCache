use reqwest_middleware::ClientWithMiddleware;
use zerocache_ports::{EmbeddingProvider, ProviderError, ProviderUsage};

use crate::client::{build_client, KIT_VERSION};
use crate::strategy::CloudRouter;

/// The single `EmbeddingProvider` implementation shared by every cloud
/// adapter. Owns client construction, chunking, sending, HTTP status mapping,
/// response-count checking, and usage accumulation -- everything the four
/// wire-shape-fixed adapters each carry their own copy of.
pub struct CloudProvider<R: CloudRouter> {
    client: ClientWithMiddleware,
    router: R,
    version: &'static str,
}

impl<R: CloudRouter> CloudProvider<R> {
    /// `version` is the *cloud crate's* own `env!("CARGO_PKG_VERSION")`, not
    /// the kit's, preserving the existing convention that cache-key
    /// versioning tracks each adapter crate's own Cargo.toml. The kit's
    /// version is folded into `cache_scope` separately.
    pub fn new(router: R, version: &'static str) -> Self {
        Self { client: build_client(), router, version }
    }

    pub fn router(&self) -> &R {
        &self.router
    }
}

#[async_trait::async_trait]
impl<R: CloudRouter + 'static> EmbeddingProvider for CloudProvider<R> {
    async fn embed_batch(
        &self,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError> {
        let resolved = self.router.resolve(model)?;
        let strategy = self.router.strategy_for(&resolved)?;

        let max_batch = strategy.max_batch();
        if max_batch == 0 {
            return Err(ProviderError("strategy reported a max batch size of 0".to_string()));
        }

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut prompt_tokens: u32 = 0;
        let mut total_tokens: u32 = 0;

        for chunk in texts.chunks(max_batch) {
            let call = strategy.build_call(api_key, &resolved, chunk)?;

            let mut request = self
                .client
                .post(&call.url)
                .header("Content-Type", "application/json")
                .body(call.body);
            for (name, value) in &call.headers {
                request = request.header(*name, value);
            }

            let body = request
                .send()
                .await
                .map_err(|e| ProviderError(e.to_string()))?
                .error_for_status()
                .map_err(|e| ProviderError(e.to_string()))?
                .bytes()
                .await
                .map_err(|e| ProviderError(e.to_string()))?;

            let outcome = strategy.parse_response(chunk.len(), &body)?;

            if outcome.vectors.len() != chunk.len() {
                return Err(ProviderError(format!(
                    "expected {} embeddings in response, got {}",
                    chunk.len(),
                    outcome.vectors.len()
                )));
            }

            vectors.extend(outcome.vectors);
            // saturating rather than wrapping: an absurd token count from a
            // misbehaving upstream must not silently wrap the counter to a
            // small number and understate what was billed.
            prompt_tokens = prompt_tokens.saturating_add(outcome.usage.prompt_tokens);
            total_tokens = total_tokens.saturating_add(outcome.usage.total_tokens);
        }

        Ok((vectors, ProviderUsage { prompt_tokens, total_tokens }))
    }

    fn version(&self) -> &'static str {
        self.version
    }

    fn cache_scope(&self, model: &str) -> Result<String, ProviderError> {
        let resolved = self.router.resolve(model)?;
        Ok(format!("{}\0{}\0kit{}", resolved.endpoint_base, resolved.canonical, KIT_VERSION))
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use zerocache_ports::ProviderUsage;

    use super::*;
    use crate::client::MAX_RETRIES;
    use crate::strategy::{EmbedCall, EmbedOutcome, ResolvedModel, TextWireStrategy};

    /// Sends `{"n": <chunk len>}` and reads back
    /// `{"vectors": [[..], ..], "tokens": n}` -- a shape chosen to be nothing
    /// like any real cloud's, so these tests can only be exercising the
    /// driver.
    struct FakeStrategy {
        max_batch: usize,
    }

    impl TextWireStrategy for FakeStrategy {
        fn max_batch(&self) -> usize {
            self.max_batch
        }

        fn build_call(
            &self,
            api_key: &str,
            resolved: &ResolvedModel,
            texts: &[String],
        ) -> Result<EmbedCall, ProviderError> {
            Ok(EmbedCall {
                url: format!("{}/fake/{}", resolved.endpoint_base, resolved.model_id),
                headers: vec![("Authorization", format!("Bearer {api_key}"))],
                body: serde_json::to_vec(&serde_json::json!({ "n": texts.len() }))
                    .map_err(|e| ProviderError(e.to_string()))?,
            })
        }

        fn parse_response(&self, _expected: usize, body: &[u8]) -> Result<EmbedOutcome, ProviderError> {
            #[derive(serde::Deserialize)]
            struct Body {
                vectors: Vec<Vec<f32>>,
                tokens: u32,
            }
            let parsed: Body = serde_json::from_slice(body).map_err(|e| ProviderError(e.to_string()))?;
            Ok(EmbedOutcome {
                vectors: parsed.vectors,
                usage: ProviderUsage { prompt_tokens: parsed.tokens, total_tokens: parsed.tokens },
            })
        }
    }

    struct FakeRouter {
        endpoint_base: String,
        strategy: FakeStrategy,
    }

    impl CloudRouter for FakeRouter {
        fn resolve(&self, model: &str) -> Result<ResolvedModel, ProviderError> {
            if model.is_empty() {
                return Err(ProviderError("model must not be empty".to_string()));
            }
            Ok(ResolvedModel {
                canonical: format!("fake/{model}"),
                model_id: model.to_string(),
                endpoint_base: self.endpoint_base.clone(),
                qualifier: None,
            })
        }

        fn strategy_for(&self, _resolved: &ResolvedModel) -> Result<&dyn TextWireStrategy, ProviderError> {
            Ok(&self.strategy)
        }
    }

    fn provider(base: String, max_batch: usize) -> CloudProvider<FakeRouter> {
        CloudProvider::new(
            FakeRouter { endpoint_base: base, strategy: FakeStrategy { max_batch } },
            "test-v1",
        )
    }

    #[tokio::test]
    async fn chunks_by_strategy_max_batch_and_concatenates_in_input_order() {
        let server = MockServer::start_async().await;
        let first = server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m").json_body(serde_json::json!({ "n": 3 }));
                then.status(200)
                    .json_body(serde_json::json!({ "vectors": [[1.0], [2.0], [3.0]], "tokens": 10 }));
            })
            .await;
        let second = server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m").json_body(serde_json::json!({ "n": 2 }));
                then.status(200).json_body(serde_json::json!({ "vectors": [[4.0], [5.0]], "tokens": 7 }));
            })
            .await;

        let texts: Vec<String> = (1..=5).map(|i| format!("t{i}")).collect();
        let (vectors, usage) = provider(server.base_url(), 3)
            .embed_batch("key", "m", &texts)
            .await
            .unwrap();

        first.assert_async().await;
        second.assert_async().await;
        assert_eq!(vectors, vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]);
        assert_eq!(usage.prompt_tokens, 17, "usage must accumulate across chunks, not report only the last");
        assert_eq!(usage.total_tokens, 17);
    }

    #[tokio::test]
    async fn max_batch_of_one_sends_one_call_per_text() {
        // Not a pathological case: Bedrock's Titan takes a scalar inputText
        // and Vertex's gemini-embedding-001 accepts a single instance, so
        // this is the real behavior of two shipped strategies.
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m").json_body(serde_json::json!({ "n": 1 }));
                then.status(200).json_body(serde_json::json!({ "vectors": [[9.0]], "tokens": 2 }));
            })
            .await;

        let texts: Vec<String> = (1..=4).map(|i| format!("t{i}")).collect();
        let (vectors, usage) = provider(server.base_url(), 1)
            .embed_batch("key", "m", &texts)
            .await
            .unwrap();

        assert_eq!(mock.hits_async().await, 4);
        assert_eq!(vectors.len(), 4);
        assert_eq!(usage.prompt_tokens, 8);
    }

    #[tokio::test]
    async fn response_count_mismatch_is_a_hard_error() {
        let server = MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m");
                then.status(200).json_body(serde_json::json!({ "vectors": [[1.0]], "tokens": 1 }));
            })
            .await;

        let texts = vec!["a".to_string(), "b".to_string()];
        let result = provider(server.base_url(), 10).embed_batch("key", "m", &texts).await;

        assert!(result.is_err(), "a short response must fail loudly, never silently misalign vectors with inputs");
    }

    #[tokio::test]
    async fn forwards_strategy_headers_and_always_sets_content_type() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/fake/m")
                    .header("Authorization", "Bearer secret")
                    .header("Content-Type", "application/json");
                then.status(200).json_body(serde_json::json!({ "vectors": [[1.0]], "tokens": 1 }));
            })
            .await;

        provider(server.base_url(), 10)
            .embed_batch("secret", "m", &["a".to_string()])
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn retries_a_transient_429_then_gives_up() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m");
                then.status(429).body("throttled");
            })
            .await;

        let result = provider(server.base_url(), 10)
            .embed_batch("key", "m", &["a".to_string()])
            .await;

        assert!(result.is_err(), "must still fail once retries are exhausted, not hang or silently succeed");
        assert_eq!(
            mock.hits_async().await,
            (MAX_RETRIES + 1) as usize,
            "1 initial attempt + MAX_RETRIES retries -- proves retry actually happened, not just that failure was reported"
        );
    }

    #[tokio::test]
    async fn does_not_retry_a_fatal_403() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/fake/m");
                then.status(403).body("denied");
            })
            .await;

        let result = provider(server.base_url(), 10)
            .embed_batch("bad", "m", &["a".to_string()])
            .await;

        assert!(result.is_err());
        assert_eq!(
            mock.hits_async().await,
            1,
            "a 403 will never succeed on retry -- retrying it would only slow down a real auth failure"
        );
    }

    #[tokio::test]
    async fn a_router_resolution_failure_surfaces_before_any_http_call() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(POST);
                then.status(200).body("{}");
            })
            .await;

        let result = provider(server.base_url(), 10).embed_batch("key", "", &["a".to_string()]).await;

        assert!(result.is_err());
        assert_eq!(mock.hits_async().await, 0, "an unresolvable model must not reach the network");
    }

    #[test]
    fn cache_scope_differs_for_two_different_models_and_carries_the_kit_version() {
        let p = provider("https://example.invalid".to_string(), 10);
        let a = p.cache_scope("model-a").unwrap();
        let b = p.cache_scope("model-b").unwrap();
        assert_ne!(a, b);
        assert!(a.contains(KIT_VERSION), "a kit behavior change must invalidate cloud cache entries");
        assert!(a.contains("https://example.invalid"));
    }

    #[test]
    fn cache_scope_propagates_a_resolution_error_rather_than_inventing_a_scope() {
        let p = provider("https://example.invalid".to_string(), 10);
        assert!(p.cache_scope("").is_err());
    }
}

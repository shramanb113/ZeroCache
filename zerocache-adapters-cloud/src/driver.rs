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
        Ok(format!("{}|{}|kit{}", resolved.endpoint_base, resolved.canonical, KIT_VERSION))
    }
}

//! Semantic LLM completion cache -- the orchestration layer for
//! `POST /{provider}/v1/chat/completions`. Composes the pure pieces from
//! `zerocache_core` (`canonicalize_completion_request`,
//! `completion_request_is_cacheable`, `CacheKey::derive_completion`) with the
//! `CompletionStore` / `ChatCompletionProvider` ports.
//!
//! v1 scope: OpenAI `/v1/chat/completions` shape, non-streaming, Tier-1
//! canonical exact-match only. Streaming replay, a local-embedder semantic
//! tier, and Anthropic `/v1/messages` are follow-ups.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::future::FutureExt;
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use zerocache_core::{canonicalize_completion_request, completion_request_is_cacheable, CacheKey};
use zerocache_ports::{ChatCompletionProvider, ChatCompletionResponse, CompletionUsage};

use crate::app::{run_store_task, AppError, AppState, SharedCompletion, SharedCompletionOutput};

/// The stored form of a cached completion: the upstream response body plus
/// the token counts it reported, so a later hit can both replay the body and
/// tell the metrics layer exactly how many tokens the caller did not get
/// billed for.
#[derive(Debug, Serialize, Deserialize)]
struct CachedCompletion {
    body: serde_json::Value,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Everything one `POST /{provider}/v1/chat/completions` call needs.
pub struct CompletionRequest<'a> {
    pub provider: Arc<dyn ChatCompletionProvider>,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    /// The full parsed OpenAI chat-completion request body.
    pub body: &'a serde_json::Value,
}

pub struct CompletionOutcome {
    /// What to send back to the caller: the replayed cached response
    /// (status 200) on a hit, or the live upstream response (its real
    /// status) otherwise.
    pub response: ChatCompletionResponse,
    /// True when `response` came from cache with no provider call.
    pub hit: bool,
}

/// Runs the completion cache flow for one request:
///  1. If the request is not deterministic enough to cache
///     (`completion_request_is_cacheable`: temperature 0 or an explicit
///     seed, `n` absent/1), forward it upstream and return the response
///     untouched -- nothing stored, nothing counted.
///  2. Otherwise derive the completion cache key and look it up. On a hit,
///     replay the stored body (as a synthetic `200`) and record the tokens
///     the caller was not billed for.
///  3. On a miss, fetch upstream (coalescing concurrent identical misses via
///     `AppState.completion_in_flight`), store the response iff it is a
///     2xx, and record the miss.
#[tracing::instrument(skip_all, fields(provider = %request.provider_name, hit))]
pub async fn complete(
    state: &AppState,
    request: CompletionRequest<'_>,
) -> Result<CompletionOutcome, AppError> {
    if !completion_request_is_cacheable(request.body) {
        let response = request
            .provider
            .chat_completion(request.api_key, request.body)
            .await
            .map_err(AppError::Provider)?;
        return Ok(CompletionOutcome {
            response,
            hit: false,
        });
    }

    let version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let canonical = canonicalize_completion_request(request.body);
    let key = CacheKey::derive_completion(
        request.owner_id,
        request.provider_name,
        &cache_scope,
        request.model,
        version,
        &canonical,
    );

    let cached = {
        let store = Arc::clone(&state.completion_store);
        run_store_task(move || store.get(&key).map_err(AppError::Store))
            .instrument(tracing::info_span!("store_lookup"))
            .await?
    };

    // A stored record that no longer deserializes (format drift, a truncated
    // write) is treated as a miss, not a hard error: content-addressed
    // entries are never wrong, only absent, so this should degrade to
    // "call the provider", never "fail the request".
    if let Some(record) =
        cached.and_then(|bytes| serde_json::from_slice::<CachedCompletion>(&bytes).ok())
    {
        let usage = CompletionUsage {
            prompt_tokens: record.prompt_tokens,
            completion_tokens: record.completion_tokens,
            total_tokens: record.total_tokens,
        };
        state
            .metrics
            .record_completion_hit(request.provider_name, &usage);
        tracing::Span::current().record("hit", true);
        return Ok(CompletionOutcome {
            response: ChatCompletionResponse {
                status: 200,
                body: record.body,
                usage,
            },
            hit: true,
        });
    }

    tracing::Span::current().record("hit", false);
    let response = fetch_completion_coalesced(state, &request, key).await?;

    if (200..300).contains(&response.status) {
        let record = CachedCompletion {
            body: response.body.clone(),
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
        };
        let bytes = serde_json::to_vec(&record)
            .expect("a CachedCompletion built from a serde_json::Value always re-serializes");
        let store = Arc::clone(&state.completion_store);
        run_store_task(move || store.put(key, bytes).map_err(AppError::Store))
            .instrument(tracing::info_span!("store_write_back"))
            .await?;
    }

    state.metrics.record_completion_miss(request.provider_name);
    Ok(CompletionOutcome {
        response,
        hit: false,
    })
}

/// Fetches one completion from the provider, coalescing with any identical
/// concurrent in-flight fetch (`AppState.completion_in_flight`, keyed by the
/// completion `CacheKey`) so N concurrent requests missing on the same key
/// trigger one upstream call, not N. In-process only, exactly like
/// `crate::app::fetch_coalesced` for embeddings -- see its doc comment for
/// the full rationale.
async fn fetch_completion_coalesced(
    state: &AppState,
    request: &CompletionRequest<'_>,
    key: CacheKey,
) -> Result<ChatCompletionResponse, AppError> {
    enum Claim {
        Owned(SharedCompletion),
        Piggyback(SharedCompletion),
    }

    let claim = {
        let mut in_flight = state
            .completion_in_flight
            .lock()
            .expect("completion_in_flight mutex poisoned");

        if let Some(existing) = in_flight.get(&key) {
            Claim::Piggyback(existing.clone())
        } else {
            let provider = Arc::clone(&request.provider);
            let api_key = request.api_key.to_string();
            let body = request.body.clone();
            let fut: Pin<Box<dyn Future<Output = SharedCompletionOutput> + Send>> = Box::pin(
                async move {
                    let response = provider.chat_completion(&api_key, &body).await?;
                    Ok(Arc::new(response))
                }
                .instrument(tracing::info_span!("provider_call")),
            );
            let shared: SharedCompletion = fut.shared();
            in_flight.insert(key, shared.clone());
            Claim::Owned(shared)
        }
    };

    match claim {
        Claim::Owned(fut) => {
            let result = fut.await;
            // Drop the completed future from the map so a later, genuinely
            // new miss for this key starts a fresh fetch.
            state
                .completion_in_flight
                .lock()
                .expect("completion_in_flight mutex poisoned")
                .remove(&key);
            let response = result.map_err(AppError::Provider)?;
            Ok((*response).clone())
        }
        Claim::Piggyback(fut) => {
            let response = fut.await.map_err(AppError::Provider)?;
            Ok((*response).clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    use serde_json::json;
    use zerocache_core::CacheKey;
    use zerocache_ports::{CompletionStore, EmbeddingStore, ProviderError, StoreError};

    use super::*;
    use crate::app::Metrics;

    struct MockCompletionStore {
        data: Mutex<HashMap<CacheKey, Vec<u8>>>,
    }

    impl MockCompletionStore {
        fn empty() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl CompletionStore for MockCompletionStore {
        fn get(&self, key: &CacheKey) -> Result<Option<Vec<u8>>, StoreError> {
            Ok(self.data.lock().unwrap().get(key).cloned())
        }
        fn put(&self, key: CacheKey, value: Vec<u8>) -> Result<(), StoreError> {
            self.data.lock().unwrap().insert(key, value);
            Ok(())
        }
        fn delete(&self, key: &CacheKey) -> Result<(), StoreError> {
            self.data.lock().unwrap().remove(key);
            Ok(())
        }
    }

    struct NoopEmbeddingStore;

    impl EmbeddingStore for NoopEmbeddingStore {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Ok(None)
        }
        fn put(&self, _key: CacheKey, _vector: Vec<f32>) -> Result<(), StoreError> {
            Ok(())
        }
        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Ok(())
        }
    }

    struct MockChatProvider {
        calls: AtomicUsize,
        status: u16,
        body: serde_json::Value,
        usage: CompletionUsage,
        delay: Option<Duration>,
        scope: String,
    }

    impl MockChatProvider {
        fn ok(body: serde_json::Value, usage: CompletionUsage) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                status: 200,
                body,
                usage,
                delay: None,
                scope: "mock-scope".to_string(),
            }
        }
        fn with_status(status: u16) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                status,
                body: json!({"error": "upstream"}),
                usage: CompletionUsage::default(),
                delay: None,
                scope: "mock-scope".to_string(),
            }
        }
        fn slow(body: serde_json::Value, delay: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                status: 200,
                body,
                usage: CompletionUsage::default(),
                delay: Some(delay),
                scope: "mock-scope".to_string(),
            }
        }
        fn with_scope(mut self, scope: &str) -> Self {
            self.scope = scope.to_string();
            self
        }
        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ChatCompletionProvider for MockChatProvider {
        async fn chat_completion(
            &self,
            _api_key: &str,
            _request: &serde_json::Value,
        ) -> Result<ChatCompletionResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(d) = self.delay {
                tokio::time::sleep(d).await;
            }
            Ok(ChatCompletionResponse {
                status: self.status,
                body: self.body.clone(),
                usage: self.usage,
            })
        }
        fn version(&self) -> &'static str {
            "mock-chat-v1"
        }
        fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
            Ok(self.scope.clone())
        }
    }

    const OWNER_A: [u8; 32] = [1u8; 32];
    const OWNER_B: [u8; 32] = [2u8; 32];

    fn state(store: impl CompletionStore + 'static) -> AppState {
        AppState {
            store: Arc::new(NoopEmbeddingStore),
            providers: HashMap::new(),
            image_providers: HashMap::new(),
            metrics: Metrics::new(),
            in_flight: Mutex::new(HashMap::new()),
            image_in_flight: Mutex::new(HashMap::new()),
            completion_store: Arc::new(store),
            completion_providers: HashMap::new(),
            completion_in_flight: Mutex::new(HashMap::new()),
        }
    }

    fn req<'a>(
        provider: &Arc<MockChatProvider>,
        owner: [u8; 32],
        body: &'a serde_json::Value,
    ) -> CompletionRequest<'a> {
        CompletionRequest {
            provider: Arc::clone(provider) as Arc<dyn ChatCompletionProvider>,
            provider_name: "openai",
            api_key: "sk-caller",
            owner_id: owner,
            model: "gpt-4o",
            body,
        }
    }

    fn eligible_body() -> serde_json::Value {
        json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "temperature": 0})
    }

    #[tokio::test]
    async fn an_ineligible_request_is_forwarded_and_never_cached() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"ok": true}),
            CompletionUsage::default(),
        ));
        let st = state(MockCompletionStore::empty());
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7
        });

        let out = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert!(!out.hit);
        assert_eq!(provider.call_count(), 1);

        complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert_eq!(
            provider.call_count(),
            2,
            "a non-deterministic request must never be served from cache"
        );
    }

    #[tokio::test]
    async fn a_first_eligible_request_misses_calls_the_provider_and_is_stored() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"choices": [{"message": {"content": "hello"}}]}),
            CompletionUsage {
                prompt_tokens: 40,
                completion_tokens: 12,
                total_tokens: 52,
            },
        ));
        let st = state(MockCompletionStore::empty());
        let body = eligible_body();

        let out = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert!(!out.hit);
        assert_eq!(out.response.status, 200);
        assert_eq!(provider.call_count(), 1);

        let out2 = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert!(out2.hit, "an identical repeat request must be a cache hit");
        assert_eq!(
            provider.call_count(),
            1,
            "the hit must not call the provider"
        );
        assert_eq!(
            out2.response.body,
            json!({"choices": [{"message": {"content": "hello"}}]})
        );
    }

    #[tokio::test]
    async fn a_non_2xx_upstream_response_is_returned_but_not_cached() {
        let provider = Arc::new(MockChatProvider::with_status(429));
        let st = state(MockCompletionStore::empty());
        let body = eligible_body();

        let out = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert_eq!(out.response.status, 429);
        assert!(!out.hit);

        complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert_eq!(
            provider.call_count(),
            2,
            "an error response must not be cached"
        );
    }

    #[tokio::test]
    async fn the_completion_cache_is_scoped_per_owner() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"x": 1}),
            CompletionUsage::default(),
        ));
        let st = state(MockCompletionStore::empty());
        let body = eligible_body();

        complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        complete(&st, req(&provider, OWNER_B, &body)).await.unwrap();
        assert_eq!(
            provider.call_count(),
            2,
            "a different caller must not hit another caller's cached completion"
        );
    }

    #[tokio::test]
    async fn two_providers_with_different_cache_scopes_do_not_share_stored_completions() {
        let st = state(MockCompletionStore::empty());
        let body = eligible_body();

        let p_a = Arc::new(
            MockChatProvider::ok(
                json!({"choices": [{"message": {"role": "assistant", "content": "A"}}]}),
                CompletionUsage::default(),
            )
            .with_scope("https://api.openai.com/v1"),
        );
        let p_b = Arc::new(
            MockChatProvider::ok(
                json!({"choices": [{"message": {"role": "assistant", "content": "B"}}]}),
                CompletionUsage::default(),
            )
            .with_scope("https://generativelanguage.googleapis.com/v1beta/openai"),
        );

        let out_a = complete(&st, req(&p_a, OWNER_A, &body)).await.unwrap();
        assert!(!out_a.hit, "first call is a miss");

        let out_b = complete(&st, req(&p_b, OWNER_A, &body)).await.unwrap();
        assert!(
            !out_b.hit,
            "a different cache_scope must not reuse another endpoint's stored completion"
        );
        assert_eq!(
            out_b.response.body["choices"][0]["message"]["content"],
            "B"
        );
    }

    #[tokio::test]
    async fn a_change_to_a_key_irrelevant_field_still_hits() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"x": 1}),
            CompletionUsage::default(),
        ));
        let st = state(MockCompletionStore::empty());

        let first = eligible_body();
        complete(&st, req(&provider, OWNER_A, &first))
            .await
            .unwrap();

        let second = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
            "user": "bob",
            "stream": false
        });
        let out = complete(&st, req(&provider, OWNER_A, &second))
            .await
            .unwrap();
        assert!(
            out.hit,
            "user/stream do not change the completion, so this must be a hit"
        );
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn a_change_to_a_generation_param_causes_a_miss() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"x": 1}),
            CompletionUsage::default(),
        ));
        let st = state(MockCompletionStore::empty());

        let a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
            "max_tokens": 100
        });
        let b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
            "max_tokens": 200
        });
        complete(&st, req(&provider, OWNER_A, &a)).await.unwrap();
        complete(&st, req(&provider, OWNER_A, &b)).await.unwrap();
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn a_hit_records_the_tokens_the_caller_was_not_billed_for() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"x": 1}),
            CompletionUsage {
                prompt_tokens: 50,
                completion_tokens: 20,
                total_tokens: 70,
            },
        ));
        let st = state(MockCompletionStore::empty());
        let body = eligible_body();

        complete(&st, req(&provider, OWNER_A, &body)).await.unwrap(); // miss, stores usage
        complete(&st, req(&provider, OWNER_A, &body)).await.unwrap(); // hit

        let dump = st.metrics.encode();
        assert!(
            dump.contains("zerocache_completion_cache_hits_total{provider=\"openai\"} 1"),
            "{dump}"
        );
        assert!(
            dump.contains("zerocache_completion_cache_misses_total{provider=\"openai\"} 1"),
            "{dump}"
        );
        assert!(
            dump.contains("zerocache_completion_prompt_tokens_saved_total{provider=\"openai\"} 50"),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_completion_completion_tokens_saved_total{provider=\"openai\"} 20"
            ),
            "{dump}"
        );
    }

    #[tokio::test]
    async fn concurrent_identical_misses_are_coalesced_into_one_provider_call() {
        let provider = Arc::new(MockChatProvider::slow(
            json!({"x": 1}),
            Duration::from_millis(50),
        ));
        let st = Arc::new(state(MockCompletionStore::empty()));
        let body = eligible_body();

        let mut handles = Vec::new();
        for _ in 0..5 {
            let st = Arc::clone(&st);
            let provider = Arc::clone(&provider);
            let body = body.clone();
            handles.push(tokio::spawn(async move {
                complete(
                    &st,
                    CompletionRequest {
                        provider: provider as Arc<dyn ChatCompletionProvider>,
                        provider_name: "openai",
                        api_key: "sk-caller",
                        owner_id: OWNER_A,
                        model: "gpt-4o",
                        body: &body,
                    },
                )
                .await
                .unwrap()
            }));
        }

        let outs: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect();

        assert_eq!(
            provider.call_count(),
            1,
            "5 concurrent identical misses must coalesce into one provider call"
        );
        for out in &outs {
            assert_eq!(out.response.body, json!({"x": 1}));
        }
    }
}

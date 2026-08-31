//! Anthropic `/v1/messages` completion cache — the orchestration layer for
//! `POST` / `DELETE /{provider}/v1/messages`. The native counterpart to
//! `crate::completion` for callers using Anthropic's own wire shape.
//!
//! v1 scope: exact-match only. No semantic tier, no streaming buffer-and-
//! replay (a `stream: true` request is a raw passthrough handled in main.rs).
//! Reuses the `CompletionStore` port, the `CachedCompletion` record, the
//! `zerocache_completion_*` metrics, the shared `completion_in_flight` map,
//! and `coalesce_cross_replica` — all unchanged.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::future::FutureExt;
use tracing::Instrument;

use zerocache_adapters_anthropic::DEFAULT_ANTHROPIC_VERSION;
use zerocache_core::{canonicalize_messages_request, messages_request_is_cacheable, CacheKey};
use zerocache_ports::{ChatCompletionResponse, MessageHeaders, MessagesProvider, ProviderError};

use crate::app::{run_store_task, AppError, AppState, SharedCompletion, SharedCompletionOutput};
use crate::coalesce::{coalesce_cross_replica, CoalesceTiming, Coalesced, CrossReplica};
use crate::completion::{encode_cached, CachedCompletion, CompletionOutcome, HitKind};

/// Everything one `POST /{provider}/v1/messages` call needs.
pub struct MessagesRequest<'a> {
    pub provider: Arc<dyn MessagesProvider>,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    /// The full parsed Anthropic `/v1/messages` request body.
    pub body: &'a serde_json::Value,
    pub headers: MessageHeaders,
}

/// Everything a `DELETE /{provider}/v1/messages` call needs. Only the
/// output-affecting fields matter — the same ones the cache key is built from
/// — so a matching `POST` body and this request resolve to one key.
pub struct MessagesDeleteRequest<'a> {
    pub provider: Arc<dyn MessagesProvider>,
    pub provider_name: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub body: &'a serde_json::Value,
    pub headers: MessageHeaders,
}

/// Canonicalization + `anthropic-version` + `anthropic-beta` header fold +
/// `CacheKey::derive_messages` in one place so `complete_messages` and
/// `delete_messages` cannot drift on the key. The caller resolves `version`
/// and `cache_scope` from its provider handle first.
fn derive_messages_key(
    owner_id: [u8; 32],
    provider_name: &str,
    cache_scope: &str,
    model: &str,
    version: &str,
    body: &serde_json::Value,
    headers: &MessageHeaders,
) -> CacheKey {
    let mut canonical = canonicalize_messages_request(body);
    // `anthropic-version` and `anthropic-beta` both change the response shape, so
    // two requests that differ only by either must not share an entry. Folded here
    // (not in the canonicalizer) because they arrive as headers. Version is folded
    // unconditionally against the adapter's own default so an absent header and an
    // explicit "2023-06-01" collapse to one key.
    canonical.push_str("\0anthropic-version:");
    canonical.push_str(
        headers
            .anthropic_version
            .as_deref()
            .unwrap_or(DEFAULT_ANTHROPIC_VERSION),
    );
    if let Some(beta) = &headers.anthropic_beta {
        canonical.push_str("\0anthropic-beta:");
        canonical.push_str(beta);
    }
    CacheKey::derive_messages(
        owner_id,
        provider_name,
        cache_scope,
        model,
        version,
        &canonical,
    )
}

/// Runs the Messages cache flow for one request:
///  1. Not deterministic enough to cache (`messages_request_is_cacheable`:
///     `temperature == 0`, non-empty `messages`, no adaptive/enabled
///     `thinking`) -> forward upstream, return the response untouched.
///  2. Otherwise derive the key and look it up. A hit replays the stored body
///     as a synthetic `200` and records the tokens the caller was not billed.
///  3. A miss fetches upstream (coalescing concurrent identical misses via
///     `AppState.completion_in_flight`, and — for the claimer — across
///     replicas via `state.coordinator`), stores a 2xx, and records the miss.
#[tracing::instrument(skip_all, fields(provider = %request.provider_name, hit))]
pub async fn complete_messages(
    state: &AppState,
    request: MessagesRequest<'_>,
) -> Result<CompletionOutcome, AppError> {
    if !messages_request_is_cacheable(request.body) {
        let response = request
            .provider
            .messages(request.api_key, request.body, &request.headers)
            .await
            .map_err(AppError::Provider)?;
        return Ok(CompletionOutcome {
            response,
            hit: false,
            hit_kind: None,
            semantic_score: None,
        });
    }

    let version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let key = derive_messages_key(
        request.owner_id,
        request.provider_name,
        &cache_scope,
        request.model,
        version,
        request.body,
        &request.headers,
    );

    let cached = {
        let store = Arc::clone(&state.completion_store);
        run_store_task(move || store.get(&key).map_err(AppError::Store))
            .instrument(tracing::info_span!("store_lookup"))
            .await?
    };

    // A stored record that no longer deserializes is treated as a miss, not a
    // hard error — content-addressed entries are absent, never wrong.
    if let Some(record) =
        cached.and_then(|bytes| serde_json::from_slice::<CachedCompletion>(&bytes).ok())
    {
        let usage = record.usage_struct();
        state
            .metrics
            .record_completion_hit(request.provider_name, false, &usage);
        tracing::Span::current().record("hit", true);
        return Ok(CompletionOutcome {
            response: ChatCompletionResponse {
                status: 200,
                body: record.body,
                usage,
            },
            hit: true,
            hit_kind: Some(HitKind::Exact),
            semantic_score: None,
        });
    }
    tracing::Span::current().record("hit", false);

    let (response, coalesced) = fetch_messages_coalesced(state, &request, key).await?;

    if let Coalesced::FromPeer = coalesced {
        // A peer replica filled this entry while we waited — a genuine hit.
        state
            .metrics
            .record_completion_hit(request.provider_name, false, &response.usage);
        state
            .metrics
            .record_cross_replica_coalesced(request.provider_name, "completion");
        tracing::Span::current().record("hit", true);
        return Ok(CompletionOutcome {
            response,
            hit: true,
            hit_kind: Some(HitKind::Exact),
            semantic_score: None,
        });
    }

    // `Piggyback` is a miss (item 21) but the request it rode on already
    // stored this exact value — do not re-store.
    if matches!(coalesced, Coalesced::Local) && (200..300).contains(&response.status) {
        let bytes = encode_cached(&response);
        let store = Arc::clone(&state.completion_store);
        run_store_task(move || store.put(key, bytes).map_err(AppError::Store))
            .instrument(tracing::info_span!("store_write_back"))
            .await?;
    }

    state
        .metrics
        .record_completion_miss(request.provider_name, false);
    Ok(CompletionOutcome {
        response,
        hit: false,
        hit_kind: None,
        semantic_score: None,
    })
}

/// Evicts the entry a matching `POST /{provider}/v1/messages` would hit,
/// scoped to this caller's `owner_id`. Idempotent. No semantic branch (tier
/// deferred), no coordinator (a content-addressed delete is local).
#[tracing::instrument(skip_all, fields(provider = %request.provider_name))]
pub async fn delete_messages(
    state: &AppState,
    request: MessagesDeleteRequest<'_>,
) -> Result<(), AppError> {
    let version = request.provider.version();
    let cache_scope = request
        .provider
        .cache_scope(request.model)
        .map_err(AppError::Provider)?;
    let key = derive_messages_key(
        request.owner_id,
        request.provider_name,
        &cache_scope,
        request.model,
        version,
        request.body,
        &request.headers,
    );

    let store = Arc::clone(&state.completion_store);
    run_store_task(move || store.delete(&key).map_err(AppError::Store))
        .instrument(tracing::info_span!("store_delete"))
        .await?;
    Ok(())
}

/// The `/v1/messages` sibling of `crate::completion::fetch_completion_coalesced`
/// — structurally identical (in-process claim/piggyback under the shared
/// `completion_in_flight` map, cross-replica single-flight inside the shared
/// future), differing only in the upstream call it wraps
/// (`MessagesProvider::messages` with the forwarded `MessageHeaders`). Kept as
/// a parallel function, matching the repo's `fetch_coalesced` /
/// `fetch_image_coalesced` precedent, rather than generalizing the chat
/// helper's signature.
async fn fetch_messages_coalesced(
    state: &AppState,
    request: &MessagesRequest<'_>,
    key: CacheKey,
) -> Result<(ChatCompletionResponse, Coalesced), AppError> {
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
            let headers = request.headers.clone();
            let coordinator = Arc::clone(&state.coordinator);
            let completion_store = Arc::clone(&state.completion_store);

            let fut: Pin<Box<dyn Future<Output = SharedCompletionOutput> + Send>> = Box::pin(
                async move {
                    let outcome = coalesce_cross_replica::<ChatCompletionResponse, _, _>(
                        &coordinator,
                        key,
                        CoalesceTiming::PROD,
                        || {
                            let store = Arc::clone(&completion_store);
                            async move {
                                let bytes = run_store_task(move || {
                                    store.get(&key).map_err(AppError::Store)
                                })
                                .await
                                .unwrap_or(None);
                                Ok(bytes
                                    .and_then(|b| {
                                        serde_json::from_slice::<CachedCompletion>(&b).ok()
                                    })
                                    .map(|rec| ChatCompletionResponse {
                                        status: 200,
                                        usage: rec.usage_struct(),
                                        body: rec.body,
                                    }))
                            }
                        },
                        {
                            let store = Arc::clone(&completion_store);
                            move || async move {
                                let response = provider
                                    .messages(&api_key, &body, &headers)
                                    .await
                                    .map_err(AppError::Provider)?;
                                // Store before returning so the coordinator's
                                // `done` implies a readable entry for peers.
                                // The caller's write-back repeats it (an
                                // identical value) and is the one that
                                // surfaces a store error.
                                if (200..300).contains(&response.status) {
                                    let bytes = encode_cached(&response);
                                    let _ = run_store_task(move || {
                                        store.put(key, bytes).map_err(AppError::Store)
                                    })
                                    .await;
                                }
                                Ok(response)
                            }
                        },
                    )
                    .await;

                    match outcome {
                        Ok(CrossReplica::Led(resp)) => Ok((Arc::new(resp), Coalesced::Local)),
                        Ok(CrossReplica::Followed(resp)) => {
                            Ok((Arc::new(resp), Coalesced::FromPeer))
                        }
                        Err(AppError::Provider(e)) => Err(e),
                        Err(AppError::Store(e)) => Err(ProviderError(format!(
                            "store error during coalesced messages fetch: {e}"
                        ))),
                    }
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
            state
                .completion_in_flight
                .lock()
                .expect("completion_in_flight mutex poisoned")
                .remove(&key);
            let (response, coalesced) = result.map_err(AppError::Provider)?;
            Ok(((*response).clone(), coalesced))
        }
        Claim::Piggyback(fut) => {
            let (response, _) = fut.await.map_err(AppError::Provider)?;
            Ok(((*response).clone(), Coalesced::Piggyback))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use serde_json::json;
    use zerocache_core::CacheKey;
    use zerocache_ports::{CompletionStore, EmbeddingStore, StoreError};

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
        fn get(&self, _k: &CacheKey) -> Result<Option<Vec<f32>>, StoreError> {
            Ok(None)
        }
        fn put(&self, _k: CacheKey, _v: Vec<f32>) -> Result<(), StoreError> {
            Ok(())
        }
        fn delete(&self, _k: &CacheKey) -> Result<(), StoreError> {
            Ok(())
        }
    }

    struct MockMessagesProvider {
        calls: AtomicUsize,
        status: u16,
        body: serde_json::Value,
        usage: zerocache_ports::CompletionUsage,
        scope: String,
    }
    impl MockMessagesProvider {
        fn ok() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                status: 200,
                body: json!({
                    "type": "message",
                    "content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 9, "output_tokens": 3}
                }),
                usage: zerocache_ports::CompletionUsage {
                    prompt_tokens: 9,
                    completion_tokens: 3,
                    total_tokens: 12,
                },
                scope: "https://api.anthropic.com".into(),
            }
        }
        fn with_status(status: u16) -> Self {
            let mut s = Self::ok();
            s.status = status;
            s.body = json!({"type": "error", "error": {"type": "invalid_request_error"}});
            s.usage = zerocache_ports::CompletionUsage::default();
            s
        }
        fn with_scope(mut self, scope: &str) -> Self {
            self.scope = scope.into();
            self
        }
        fn count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }
    #[async_trait::async_trait]
    impl MessagesProvider for MockMessagesProvider {
        async fn messages(
            &self,
            _api_key: &str,
            _request: &serde_json::Value,
            _headers: &MessageHeaders,
        ) -> Result<ChatCompletionResponse, zerocache_ports::ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatCompletionResponse {
                status: self.status,
                body: self.body.clone(),
                usage: self.usage,
            })
        }
        async fn messages_stream_passthrough(
            &self,
            _api_key: &str,
            _request: &serde_json::Value,
            _headers: &MessageHeaders,
        ) -> Result<(u16, zerocache_ports::SseByteStream), zerocache_ports::ProviderError> {
            Ok((200, Box::pin(futures::stream::empty())))
        }
        fn version(&self) -> &'static str {
            "mock-messages-v1"
        }
        fn cache_scope(&self, _model: &str) -> Result<String, zerocache_ports::ProviderError> {
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
            completion_stream_providers: HashMap::new(),
            messages_providers: HashMap::new(),
            completion_in_flight: Mutex::new(HashMap::new()),
            coordinator: Arc::new(crate::coalesce::NoopCoordinator),
            #[cfg(feature = "semantic")]
            semantic: None,
        }
    }

    fn cacheable_body() -> serde_json::Value {
        json!({
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0
        })
    }

    fn req<'a>(
        provider: &Arc<MockMessagesProvider>,
        owner: [u8; 32],
        body: &'a serde_json::Value,
        headers: MessageHeaders,
    ) -> MessagesRequest<'a> {
        MessagesRequest {
            provider: Arc::clone(provider) as Arc<dyn MessagesProvider>,
            provider_name: "anthropic",
            api_key: "sk-caller",
            owner_id: owner,
            model: "claude-opus-4-6",
            body,
            headers,
        }
    }

    #[tokio::test]
    async fn a_non_cacheable_request_is_forwarded_and_never_stored() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = json!({
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 1
        });

        let out = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!out.hit);
        assert_eq!(p.count(), 1);

        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert_eq!(p.count(), 2, "a temperature:1 request must never be cached");
    }

    #[tokio::test]
    async fn a_cacheable_miss_calls_upstream_stores_2xx_and_the_repeat_is_a_hit() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        let miss = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!miss.hit);
        assert_eq!(p.count(), 1);

        let hit = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(hit.hit);
        assert_eq!(hit.hit_kind, Some(HitKind::Exact));
        assert_eq!(hit.response.status, 200);
        assert_eq!(hit.response.usage.total_tokens, 12);
        assert_eq!(p.count(), 1, "the hit must not call upstream");
    }

    #[tokio::test]
    async fn a_non_2xx_miss_is_forwarded_and_not_stored() {
        let p = Arc::new(MockMessagesProvider::with_status(400));
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        let first = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!first.hit);
        assert_eq!(first.response.status, 400);

        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert_eq!(p.count(), 2, "a non-2xx must never be cached");
    }

    #[tokio::test]
    async fn two_owners_with_an_identical_body_never_share_an_entry() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        let out_b = complete_messages(&st, req(&p, OWNER_B, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!out_b.hit);
        assert_eq!(p.count(), 2);
    }

    #[tokio::test]
    async fn two_cache_scopes_never_share_an_entry() {
        let a = Arc::new(MockMessagesProvider::ok().with_scope("https://api.anthropic.com"));
        let b = Arc::new(MockMessagesProvider::ok().with_scope("https://gw.internal/anthropic"));
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        complete_messages(&st, req(&a, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        let out = complete_messages(&st, req(&b, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!out.hit, "a different base URL is a different cache scope");
    }

    #[tokio::test]
    async fn a_key_irrelevant_field_still_hits_a_generation_param_misses() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());

        complete_messages(
            &st,
            req(&p, OWNER_A, &cacheable_body(), MessageHeaders::default()),
        )
        .await
        .unwrap();

        let mut with_metadata = cacheable_body();
        with_metadata["metadata"] = json!({"user_id": "abc"});
        let hit = complete_messages(
            &st,
            req(&p, OWNER_A, &with_metadata, MessageHeaders::default()),
        )
        .await
        .unwrap();
        assert!(hit.hit, "metadata is on the denylist");

        let mut more_tokens = cacheable_body();
        more_tokens["max_tokens"] = json!(128);
        let miss = complete_messages(
            &st,
            req(&p, OWNER_A, &more_tokens, MessageHeaders::default()),
        )
        .await
        .unwrap();
        assert!(!miss.hit, "max_tokens changes the response");
    }

    #[tokio::test]
    async fn a_different_anthropic_beta_header_misses() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        complete_messages(
            &st,
            req(
                &p,
                OWNER_A,
                &body,
                MessageHeaders {
                    anthropic_version: None,
                    anthropic_beta: Some("beta-a".into()),
                },
            ),
        )
        .await
        .unwrap();

        let out = complete_messages(
            &st,
            req(
                &p,
                OWNER_A,
                &body,
                MessageHeaders {
                    anthropic_version: None,
                    anthropic_beta: Some("beta-b".into()),
                },
            ),
        )
        .await
        .unwrap();
        assert!(!out.hit, "the anthropic-beta header folds into the key");
        assert_eq!(p.count(), 2);
    }

    #[tokio::test]
    async fn a_different_anthropic_version_header_misses() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        let h1 = MessageHeaders {
            anthropic_version: Some("2023-06-01".into()),
            anthropic_beta: None,
        };
        let h2 = MessageHeaders {
            anthropic_version: Some("2099-01-01".into()),
            anthropic_beta: None,
        };

        complete_messages(&st, req(&p, OWNER_A, &body, h1.clone()))
            .await
            .unwrap();
        complete_messages(&st, req(&p, OWNER_A, &body, h1))
            .await
            .unwrap();
        assert_eq!(p.count(), 1, "same version repeats -> hit");

        complete_messages(&st, req(&p, OWNER_A, &body, h2))
            .await
            .unwrap();
        assert_eq!(p.count(), 2, "a different anthropic-version must miss");
    }

    #[tokio::test]
    async fn an_absent_anthropic_version_and_the_explicit_default_share_an_entry() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        complete_messages(
            &st,
            req(
                &p,
                OWNER_A,
                &body,
                MessageHeaders {
                    anthropic_version: Some("2023-06-01".into()),
                    anthropic_beta: None,
                },
            ),
        )
        .await
        .unwrap();

        assert_eq!(p.count(), 1, "absent header == explicit 2023-06-01");
    }

    #[tokio::test]
    async fn a_messages_hit_bumps_the_completion_metric_counters() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap(); // miss, stores usage
        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap(); // hit

        let dump = st.metrics.encode();
        for line in [
            "zerocache_completion_cache_hits_total{provider=\"anthropic\",stream=\"false\"} 1",
            "zerocache_completion_cache_misses_total{provider=\"anthropic\",stream=\"false\"} 1",
            "zerocache_completion_prompt_tokens_saved_total{provider=\"anthropic\",stream=\"false\"} 9",
            "zerocache_completion_completion_tokens_saved_total{provider=\"anthropic\",stream=\"false\"} 3",
        ] {
            assert!(dump.contains(line), "missing `{line}` in:\n{dump}");
        }
    }

    #[tokio::test]
    async fn delete_ignores_stream_fields_in_the_body() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        // seed, then confirm a repeat is a hit
        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert_eq!(p.count(), 1, "the seeded entry hits");

        // a DELETE body carrying `stream: true` still targets the plain entry
        let mut streamy = cacheable_body();
        streamy["stream"] = json!(true);
        delete_messages(
            &st,
            MessagesDeleteRequest {
                provider: Arc::clone(&p) as Arc<dyn MessagesProvider>,
                provider_name: "anthropic",
                owner_id: OWNER_A,
                model: "claude-opus-4-6",
                body: &streamy,
                headers: MessageHeaders::default(),
            },
        )
        .await
        .unwrap();

        let re_miss = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!re_miss.hit);
        assert_eq!(p.count(), 2, "the stream-carrying DELETE evicted the entry");
    }

    #[tokio::test]
    async fn concurrent_identical_cacheable_misses_are_coalesced_into_one_upstream_call() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = Arc::new(state(MockCompletionStore::empty()));
        let body = cacheable_body();

        let mut handles = Vec::new();
        for _ in 0..5 {
            let st = Arc::clone(&st);
            let p = Arc::clone(&p);
            let body = body.clone();
            handles.push(tokio::spawn(async move {
                complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
                    .await
                    .unwrap()
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            p.count(),
            1,
            "5 concurrent identical misses => 1 upstream call"
        );
    }

    #[tokio::test]
    async fn delete_evicts_the_entry_and_is_idempotent_and_owner_scoped() {
        let p = Arc::new(MockMessagesProvider::ok());
        let st = state(MockCompletionStore::empty());
        let body = cacheable_body();

        // seed
        complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert_eq!(p.count(), 1);

        // a different owner's DELETE must not evict OWNER_A's entry
        delete_messages(
            &st,
            MessagesDeleteRequest {
                provider: Arc::clone(&p) as Arc<dyn MessagesProvider>,
                provider_name: "anthropic",
                owner_id: OWNER_B,
                model: "claude-opus-4-6",
                body: &body,
                headers: MessageHeaders::default(),
            },
        )
        .await
        .unwrap();
        let still_hit = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(still_hit.hit);
        assert_eq!(p.count(), 1);

        // OWNER_A's own DELETE evicts; the next POST re-misses
        delete_messages(
            &st,
            MessagesDeleteRequest {
                provider: Arc::clone(&p) as Arc<dyn MessagesProvider>,
                provider_name: "anthropic",
                owner_id: OWNER_A,
                model: "claude-opus-4-6",
                body: &body,
                headers: MessageHeaders::default(),
            },
        )
        .await
        .unwrap();
        let re_miss = complete_messages(&st, req(&p, OWNER_A, &body, MessageHeaders::default()))
            .await
            .unwrap();
        assert!(!re_miss.hit);
        assert_eq!(p.count(), 2);

        // deleting an already-absent key still succeeds
        delete_messages(
            &st,
            MessagesDeleteRequest {
                provider: Arc::clone(&p) as Arc<dyn MessagesProvider>,
                provider_name: "anthropic",
                owner_id: OWNER_A,
                model: "claude-opus-4-6",
                body: &json!({"model": "x", "messages": [{"role": "user", "content": "never seen"}], "temperature": 0}),
                headers: MessageHeaders::default(),
            },
        )
        .await
        .unwrap();
    }
}

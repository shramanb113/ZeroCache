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
use zerocache_ports::{
    ChatCompletionProvider, ChatCompletionResponse, CompletionUsage, ProviderError,
    StreamingChatCompletionProvider,
};

use crate::app::{run_store_task, AppError, AppState, SharedCompletion, SharedCompletionOutput};
use crate::coalesce::{coalesce_cross_replica, CoalesceTiming, Coalesced, CrossReplica};

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
    /// Verbatim upstream SSE payload (minus a withheld injected usage chunk).
    /// `None` for an entry filled by a non-streaming miss -- a `stream:true`
    /// hit on such an entry re-chunks `body` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    raw_sse: Option<Vec<u8>>,
}

fn encode_cached(response: &ChatCompletionResponse) -> Vec<u8> {
    let record = CachedCompletion {
        body: response.body.clone(),
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.total_tokens,
        raw_sse: None,
    };
    serde_json::to_vec(&record)
        .expect("a CachedCompletion built from a serde_json::Value always re-serializes")
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    Exact,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    Semantic,
}

pub struct CompletionOutcome {
    /// What to send back to the caller: the replayed cached response
    /// (status 200) on a hit, or the live upstream response (its real
    /// status) otherwise.
    pub response: ChatCompletionResponse,
    /// True when `response` came from cache with no provider call.
    pub hit: bool,
    /// `Some` on any hit; emitted as `X-Zerocache-Completion-Hit-Kind`.
    pub hit_kind: Option<HitKind>,
    /// `Some` only on a semantic hit; emitted as `X-Zerocache-Semantic-Score`.
    pub semantic_score: Option<f32>,
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
            hit_kind: None,
            semantic_score: None,
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

    #[cfg(feature = "semantic")]
    let mut semantic_ctx: Option<(zerocache_semantic::ScopeKey, [u8; 32], Vec<f32>)> = None;

    #[cfg(feature = "semantic")]
    if let Some(sem) = &state.semantic {
        if let Some((fuzzy_text, coarse_hash)) =
            crate::semantic::semantic_inputs(sem.match_unit, request.body)
        {
            let scope = crate::semantic::scope_key(
                &request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
            );
            let embedder = Arc::clone(&sem.embedder);
            let qvec = match tokio::task::spawn_blocking(move || embedder.embed(&fuzzy_text))
                .await
                .expect("embed task panicked")
            {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("semantic: embed failed, falling back to provider: {e}");
                    None
                }
            };

            if let Some(qvec) = qvec {
                if let Some(hit) = crate::semantic::semantic_probe(
                    sem,
                    &state.completion_store,
                    scope,
                    coarse_hash,
                    &qvec,
                )
                .await?
                {
                    if let Ok(record) =
                        serde_json::from_slice::<CachedCompletion>(&hit.record_bytes)
                    {
                        let usage = CompletionUsage {
                            prompt_tokens: record.prompt_tokens,
                            completion_tokens: record.completion_tokens,
                            total_tokens: record.total_tokens,
                        };
                        state
                            .metrics
                            .record_completion_hit(request.provider_name, false, &usage);
                        state
                            .metrics
                            .record_completion_semantic_hit(request.provider_name, false);
                        tracing::Span::current().record("hit", true);
                        return Ok(CompletionOutcome {
                            response: ChatCompletionResponse {
                                status: 200,
                                body: record.body,
                                usage,
                            },
                            hit: true,
                            hit_kind: Some(HitKind::Semantic),
                            semantic_score: Some(hit.score),
                        });
                    }
                }
                semantic_ctx = Some((scope, coarse_hash, qvec));
            }
        }
    }

    let (response, coalesced) = fetch_completion_coalesced(state, &request, key).await?;

    if let Coalesced::FromPeer = coalesced {
        // A peer replica filled this entry while we waited: a genuine cache
        // hit, just very fresh. Record tokens saved; do not re-store.
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

    if (200..300).contains(&response.status) {
        let bytes = encode_cached(&response);
        let store = Arc::clone(&state.completion_store);
        run_store_task(move || store.put(key, bytes).map_err(AppError::Store))
            .instrument(tracing::info_span!("store_write_back"))
            .await?;

        #[cfg(feature = "semantic")]
        if let (Some(sem), Some((scope, coarse_hash, qvec))) =
            (&state.semantic, semantic_ctx.take())
        {
            crate::semantic::record_vector(sem, key, scope, coarse_hash, qvec).await;
        }
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

/// Fetches one completion, coalescing with any identical concurrent in-flight
/// fetch on this replica (`AppState.completion_in_flight`) and -- for the
/// claiming request, inside the shared future so in-process piggybackers
/// benefit too -- across replicas via `state.coordinator`
/// (`crate::coalesce::coalesce_cross_replica`). Returns the response plus a
/// `Coalesced` marker: `FromPeer` only when this replica read the value back
/// from the store after a peer filled it. An in-process piggybacker is always
/// `Local` (it counts as a miss, item 21).
async fn fetch_completion_coalesced(
    state: &AppState,
    request: &CompletionRequest<'_>,
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
                                        body: rec.body,
                                        usage: CompletionUsage {
                                            prompt_tokens: rec.prompt_tokens,
                                            completion_tokens: rec.completion_tokens,
                                            total_tokens: rec.total_tokens,
                                        },
                                    }))
                            }
                        },
                        {
                            let store = Arc::clone(&completion_store);
                            move || async move {
                                let response = provider
                                    .chat_completion(&api_key, &body)
                                    .await
                                    .map_err(AppError::Provider)?;
                                // Store before returning, so the coordinator's
                                // `done` signal implies a readable entry for
                                // peers. The caller's write-back repeats it --
                                // an identical content-addressed value, and it
                                // is the one that surfaces a store error.
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
                        // Defensive: the read closure maps store errors to
                        // Ok(None), so this is unreachable today.
                        Err(AppError::Store(e)) => Err(ProviderError(format!(
                            "store error during coalesced completion fetch: {e}"
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
            // Drop the completed future from the map so a later, genuinely
            // new miss for this key starts a fresh fetch.
            state
                .completion_in_flight
                .lock()
                .expect("completion_in_flight mutex poisoned")
                .remove(&key);
            let (response, coalesced) = result.map_err(AppError::Provider)?;
            Ok(((*response).clone(), coalesced))
        }
        Claim::Piggyback(fut) => {
            // An in-process piggybacker counts as a miss (item 21) regardless
            // of how the claim itself resolved.
            let (response, _) = fut.await.map_err(AppError::Provider)?;
            Ok(((*response).clone(), Coalesced::Local))
        }
    }
}

// --- streaming completion cache (see crate::sse) ---

/// Everything one `stream:true` `POST /{provider}/v1/chat/completions` call
/// needs. The streaming counterpart to `CompletionRequest`.
pub struct CompletionStreamRequest<'a> {
    pub stream_provider: Arc<dyn StreamingChatCompletionProvider>,
    pub provider_name: &'a str,
    pub api_key: &'a str,
    pub owner_id: [u8; 32],
    pub model: &'a str,
    pub body: &'a serde_json::Value,
}

/// What `complete_streaming` resolves to. `Stream` carries a live SSE body
/// (a tee of the upstream on a miss, a paced replay on a hit); `UpstreamError`
/// is a non-2xx upstream response, forwarded verbatim and never cached.
pub enum StreamingOutcome {
    Stream {
        body: axum::body::Body,
        hit: bool,
        hit_kind: Option<HitKind>,
        semantic_score: Option<f32>,
    },
    UpstreamError(ChatCompletionResponse),
}

/// The streaming counterpart to `complete()`. Mirrors its exact-match and
/// (feature-gated) semantic-probe lookup blocks verbatim; on a hit it replays
/// the stored completion as a paced SSE stream, and on a miss it tees the
/// upstream SSE stream to the client live while a background task
/// (`spawn_tee`) buffers it and, on a clean finish, stores both the assembled
/// non-streaming body and the raw SSE bytes.
///
/// Takes `Arc<AppState>` (not `&AppState`) because the miss path spawns a
/// background tee that outlives this call.
#[tracing::instrument(skip_all, fields(provider = %request.provider_name))]
pub async fn complete_streaming(
    state: Arc<AppState>,
    request: CompletionStreamRequest<'_>,
) -> Result<StreamingOutcome, AppError> {
    let version = request.stream_provider.version();
    let cache_scope = request
        .stream_provider
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

    // A stored record that no longer deserializes is a miss, not an error --
    // same posture as `complete()`.
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
            .record_completion_hit(request.provider_name, true, &usage);
        return Ok(StreamingOutcome::Stream {
            body: replay_body(record),
            hit: true,
            hit_kind: Some(HitKind::Exact),
            semantic_score: None,
        });
    }

    #[cfg(feature = "semantic")]
    let mut semantic_ctx: Option<(zerocache_semantic::ScopeKey, [u8; 32], Vec<f32>)> = None;

    #[cfg(feature = "semantic")]
    if let Some(sem) = &state.semantic {
        if let Some((fuzzy_text, coarse_hash)) =
            crate::semantic::semantic_inputs(sem.match_unit, request.body)
        {
            let scope = crate::semantic::scope_key(
                &request.owner_id,
                request.provider_name,
                &cache_scope,
                request.model,
            );
            let embedder = Arc::clone(&sem.embedder);
            let qvec = match tokio::task::spawn_blocking(move || embedder.embed(&fuzzy_text))
                .await
                .expect("embed task panicked")
            {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("semantic: embed failed, falling back to provider: {e}");
                    None
                }
            };

            if let Some(qvec) = qvec {
                if let Some(hit) = crate::semantic::semantic_probe(
                    sem,
                    &state.completion_store,
                    scope,
                    coarse_hash,
                    &qvec,
                )
                .await?
                {
                    if let Ok(record) =
                        serde_json::from_slice::<CachedCompletion>(&hit.record_bytes)
                    {
                        let usage = CompletionUsage {
                            prompt_tokens: record.prompt_tokens,
                            completion_tokens: record.completion_tokens,
                            total_tokens: record.total_tokens,
                        };
                        state
                            .metrics
                            .record_completion_hit(request.provider_name, true, &usage);
                        state
                            .metrics
                            .record_completion_semantic_hit(request.provider_name, true);
                        return Ok(StreamingOutcome::Stream {
                            body: replay_body(record),
                            hit: true,
                            hit_kind: Some(HitKind::Semantic),
                            semantic_score: Some(hit.score),
                        });
                    }
                }
                semantic_ctx = Some((scope, coarse_hash, qvec));
            }
        }
    }

    // Miss: open the upstream stream, injecting stream_options.include_usage
    // when the caller did not ask for it (so the stored record carries real
    // token counts even if the client stream must not show them).
    let patched = patch_stream_options(request.body);
    let injected = patched != *request.body;

    let (status, mut upstream) = request
        .stream_provider
        .chat_completion_stream(request.api_key, &patched)
        .await
        .map_err(AppError::Provider)?;

    if !(200..300).contains(&status) {
        use futures::StreamExt;
        let mut buf = Vec::new();
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => buf.extend_from_slice(&chunk),
                Err(_) => break,
            }
        }
        let body = serde_json::from_slice::<serde_json::Value>(&buf).unwrap_or_else(
            |_| serde_json::json!({ "error": String::from_utf8_lossy(&buf).into_owned() }),
        );
        return Ok(StreamingOutcome::UpstreamError(ChatCompletionResponse {
            status,
            body,
            usage: CompletionUsage::default(),
        }));
    }

    let (tx, rx) = futures::channel::mpsc::unbounded::<Result<axum::body::Bytes, std::io::Error>>();
    spawn_tee(
        Arc::clone(&state),
        request.provider_name.to_string(),
        key,
        upstream,
        tx,
        injected,
        #[cfg(feature = "semantic")]
        semantic_ctx.take(),
    );

    Ok(StreamingOutcome::Stream {
        body: axum::body::Body::from_stream(rx),
        hit: false,
        hit_kind: None,
        semantic_score: None,
    })
}

/// True when `ev` is the trailing usage-only chunk OpenAI emits when
/// `stream_options.include_usage` is set: `Data` with an empty `choices`
/// array and a `usage` key.
fn is_usage_only_data(ev: &crate::sse::SseEvent) -> bool {
    match ev {
        crate::sse::SseEvent::Data(v) => {
            v.get("choices")
                .and_then(|c| c.as_array())
                .is_some_and(|a| a.is_empty())
                && v.get("usage").is_some()
        }
        _ => false,
    }
}

/// Background task: read the upstream SSE stream to the end, forwarding each
/// chunk to the client (`tx`) while assembling the non-streaming body and
/// capturing the raw bytes. On a clean finish (`Completeness::Complete`),
/// store `{assembled body, token counts, raw SSE}`. An incomplete or
/// error-carrying stream stores nothing.
#[allow(clippy::too_many_arguments)]
fn spawn_tee(
    state: Arc<AppState>,
    provider_name: String,
    key: CacheKey,
    mut upstream: zerocache_ports::SseByteStream,
    tx: futures::channel::mpsc::UnboundedSender<Result<axum::body::Bytes, std::io::Error>>,
    injected: bool,
    #[cfg(feature = "semantic")] semantic_ctx: Option<(
        zerocache_semantic::ScopeKey,
        [u8; 32],
        Vec<f32>,
    )>,
) {
    tokio::spawn(async move {
        use futures::StreamExt;

        use crate::sse::{Completeness, DeltaAssembler, SseFrameParser};

        let mut parser = SseFrameParser::new();
        let mut asm = DeltaAssembler::new();
        let mut raw_capture: Vec<u8> = Vec::new();
        let mut malformed: Vec<String> = Vec::new();
        let mut client_gone = false;

        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    let events = parser.feed(&chunk);
                    let suppress = injected && events.len() == 1 && is_usage_only_data(&events[0]);
                    for ev in &events {
                        asm.ingest(ev);
                        if let crate::sse::SseEvent::Malformed(payload) = ev {
                            malformed.push(payload.clone());
                        }
                    }
                    if !suppress {
                        if !client_gone
                            && tx
                                .unbounded_send(Ok(axum::body::Bytes::copy_from_slice(&chunk)))
                                .is_err()
                        {
                            // Client hung up: stop forwarding, keep draining so
                            // the store still fills.
                            client_gone = true;
                        }
                        raw_capture.extend_from_slice(&chunk);
                    }
                }
                Err(e) => {
                    tracing::warn!("streamed completion upstream error, not caching: {e}");
                    break;
                }
            }
        }

        for ev in parser.finish() {
            if let crate::sse::SseEvent::Malformed(payload) = &ev {
                malformed.push(payload.clone());
            }
            asm.ingest(&ev);
        }

        let out = asm.finish();
        match out.completeness {
            Completeness::Complete => {
                let record = CachedCompletion {
                    body: out.body.clone(),
                    prompt_tokens: out.usage.prompt_tokens,
                    completion_tokens: out.usage.completion_tokens,
                    total_tokens: out.usage.total_tokens,
                    raw_sse: Some(raw_capture),
                };
                match serde_json::to_vec(&record) {
                    Ok(bytes) => {
                        let store = Arc::clone(&state.completion_store);
                        match run_store_task(move || store.put(key, bytes).map_err(AppError::Store))
                            .instrument(tracing::info_span!("store_write_back"))
                            .await
                        {
                            Ok(()) => {
                                #[cfg(feature = "semantic")]
                                if let (Some(sem), Some((scope, coarse_hash, qvec))) =
                                    (&state.semantic, semantic_ctx)
                                {
                                    crate::semantic::record_vector(
                                        sem,
                                        key,
                                        scope,
                                        coarse_hash,
                                        qvec,
                                    )
                                    .await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("streamed completion store write failed: {e}")
                            }
                        }
                    }
                    Err(e) => tracing::warn!("streamed completion serialize failed: {e}"),
                }
                state.metrics.record_completion_miss(&provider_name, true);
            }
            Completeness::Incomplete(reason) => {
                if malformed.is_empty() {
                    tracing::warn!("streamed completion not cached: {reason}");
                } else {
                    tracing::warn!(
                        "streamed completion not cached: {reason} (malformed frames: {malformed:?})"
                    );
                }
                state.metrics.record_completion_miss(&provider_name, true);
            }
        }

        // OpenAI forwards `data: [DONE]` as a normal frame, so it is already in
        // the client stream / raw_capture -- do not synthesize another here.
        drop(tx);
    });
}

/// A clone of `body` with `stream_options.include_usage = true` merged in when
/// it is not already set. Given verbatim by the task brief.
fn patch_stream_options(body: &serde_json::Value) -> serde_json::Value {
    let mut b = body.clone();
    let already = b
        .get("stream_options")
        .and_then(|so| so.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !already {
        let obj = b.as_object_mut().expect("chat body is a JSON object");
        let so = obj
            .entry("stream_options")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(so) = so.as_object_mut() {
            so.insert("include_usage".into(), serde_json::json!(true));
        }
    }
    b
}

/// Replay a stored completion as a paced SSE stream: the verbatim raw SSE
/// bytes when present, else re-chunked from the assembled body. Given
/// verbatim by the task brief.
fn replay_body(record: CachedCompletion) -> axum::body::Body {
    let frames = match &record.raw_sse {
        Some(raw) => crate::sse::split_frames(raw),
        None => crate::sse::rechunk(&record.body),
    };
    let (tx, rx) = futures::channel::mpsc::unbounded::<Result<axum::body::Bytes, std::io::Error>>();
    tokio::spawn(async move {
        let ends_with_done = frames
            .last()
            .map(|f| f.windows(6).any(|w| w == b"[DONE]"))
            .unwrap_or(false);
        for frame in frames {
            if tx
                .unbounded_send(Ok(axum::body::Bytes::from(frame)))
                .is_err()
            {
                return;
            }
            tokio::time::sleep(crate::sse::SSE_REPLAY_FRAME_DELAY).await;
        }
        if !ends_with_done {
            let _ = tx.unbounded_send(Ok(axum::body::Bytes::from_static(b"data: [DONE]\n\n")));
        }
    });
    axum::body::Body::from_stream(rx)
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

    #[test]
    fn a_pre_streaming_record_without_raw_sse_still_deserializes() {
        let legacy = serde_json::json!({
            "body": {"choices": []}, "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3
        });
        let bytes = serde_json::to_vec(&legacy).unwrap();
        let rec: CachedCompletion = serde_json::from_slice(&bytes).unwrap();
        assert!(rec.raw_sse.is_none());
        assert_eq!(rec.total_tokens, 3);
    }

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
        state_with_coordinator(store, Arc::new(crate::coalesce::NoopCoordinator))
    }

    fn state_with_coordinator(
        store: impl CompletionStore + 'static,
        coordinator: Arc<dyn zerocache_ports::CoalescingCoordinator>,
    ) -> AppState {
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
            completion_in_flight: Mutex::new(HashMap::new()),
            coordinator,
            #[cfg(feature = "semantic")]
            semantic: None,
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
    async fn a_non_cacheable_streaming_request_is_piped_through_untouched() {
        // temperature 0.7 => not cacheable => the handler must passthrough,
        // storing nothing. The end-to-end passthrough assertion lives in the
        // main.rs `stream_passthrough` test; here we pin the cacheability-gate
        // classification the handler branch relies on.
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "stream": true
        });
        assert!(!zerocache_core::completion_request_is_cacheable(&body));
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
        assert_eq!(out_b.response.body["choices"][0]["message"]["content"], "B");
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
            dump.contains(
                "zerocache_completion_cache_hits_total{provider=\"openai\",stream=\"false\"} 1"
            ),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_completion_cache_misses_total{provider=\"openai\",stream=\"false\"} 1"
            ),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_completion_prompt_tokens_saved_total{provider=\"openai\",stream=\"false\"} 50"
            ),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_completion_completion_tokens_saved_total{provider=\"openai\",stream=\"false\"} 20"
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

    struct FollowerCoordinator;
    impl zerocache_ports::CoalescingCoordinator for FollowerCoordinator {
        fn try_lead(&self, _k: &CacheKey) -> zerocache_ports::Role {
            zerocache_ports::Role::Follower
        }
        fn complete(&self, _k: &CacheKey) {}
        fn follow(&self, _k: &CacheKey, _w: Duration) -> zerocache_ports::FollowSignal {
            zerocache_ports::FollowSignal::Signalled
        }
    }

    /// Misses the exact-match lookup (`get` call #1), then hits every later
    /// `get` -- simulating a peer replica writing the entry mid-follow.
    struct PeerFillsAfterFirstGet {
        record: Vec<u8>,
        gets: AtomicUsize,
    }

    impl PeerFillsAfterFirstGet {
        fn with_body(body: serde_json::Value) -> Self {
            let rec = json!({
                "body": body, "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14
            });
            Self {
                record: serde_json::to_vec(&rec).unwrap(),
                gets: AtomicUsize::new(0),
            }
        }
    }

    impl CompletionStore for PeerFillsAfterFirstGet {
        fn get(&self, _key: &CacheKey) -> Result<Option<Vec<u8>>, StoreError> {
            let n = self.gets.fetch_add(1, Ordering::SeqCst);
            Ok(if n == 0 {
                None
            } else {
                Some(self.record.clone())
            })
        }
        fn put(&self, _key: CacheKey, _value: Vec<u8>) -> Result<(), StoreError> {
            Ok(())
        }
        fn delete(&self, _key: &CacheKey) -> Result<(), StoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_peer_filled_completion_is_served_as_a_hit_without_calling_the_provider() {
        let provider = Arc::new(MockChatProvider::ok(
            json!({"never": "called"}),
            CompletionUsage::default(),
        ));
        let body = eligible_body();
        let store = PeerFillsAfterFirstGet::with_body(json!({"peer": "value"}));

        let st = state_with_coordinator(store, Arc::new(FollowerCoordinator));
        let out = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();

        assert!(out.hit, "a peer-filled entry is a hit");
        assert_eq!(out.hit_kind, Some(HitKind::Exact));
        assert_eq!(out.response.body, json!({"peer": "value"}));
        assert_eq!(provider.call_count(), 0, "no upstream call for a peer fill");

        let dump = st.metrics.encode();
        assert!(
            dump.contains(
                "zerocache_cross_replica_coalesced_total{kind=\"completion\",provider=\"openai\"} 1"
            ),
            "{dump}"
        );
        assert!(
            dump.contains(
                "zerocache_completion_cache_hits_total{provider=\"openai\",stream=\"false\"} 1"
            ),
            "{dump}"
        );
    }

    #[tokio::test]
    async fn noop_coordinator_leaves_the_completion_miss_store_hit_cycle_unchanged() {
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
        assert_eq!(provider.call_count(), 1);
        let out2 = complete(&st, req(&provider, OWNER_A, &body)).await.unwrap();
        assert!(out2.hit);
        assert_eq!(provider.call_count(), 1);
    }

    mod streaming {
        use std::sync::atomic::AtomicBool;

        use super::*;

        struct MockStreamProvider {
            frames: Vec<Vec<u8>>,
            status: u16,
            error_body: Vec<u8>,
            calls: AtomicUsize,
            last_include_usage: AtomicBool,
        }

        impl MockStreamProvider {
            fn ok(frames: Vec<Vec<u8>>) -> Self {
                Self {
                    frames,
                    status: 200,
                    error_body: Vec::new(),
                    calls: AtomicUsize::new(0),
                    last_include_usage: AtomicBool::new(false),
                }
            }
            fn status(status: u16, body: Vec<u8>) -> Self {
                Self {
                    frames: Vec::new(),
                    status,
                    error_body: body,
                    calls: AtomicUsize::new(0),
                    last_include_usage: AtomicBool::new(false),
                }
            }
            fn calls(&self) -> usize {
                self.calls.load(Ordering::SeqCst)
            }
            fn last_request_had_include_usage(&self) -> bool {
                self.last_include_usage.load(Ordering::SeqCst)
            }
        }

        #[async_trait::async_trait]
        impl zerocache_ports::StreamingChatCompletionProvider for MockStreamProvider {
            async fn chat_completion_stream(
                &self,
                _api_key: &str,
                request: &serde_json::Value,
            ) -> Result<(u16, zerocache_ports::SseByteStream), ProviderError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let inc = request
                    .get("stream_options")
                    .and_then(|so| so.get("include_usage"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.last_include_usage.store(inc, Ordering::SeqCst);
                let items: Vec<Result<Vec<u8>, ProviderError>> =
                    if (200..300).contains(&self.status) {
                        self.frames.iter().cloned().map(Ok).collect()
                    } else {
                        vec![Ok(self.error_body.clone())]
                    };
                Ok((self.status, Box::pin(futures::stream::iter(items))))
            }
            // Must match MockChatProvider's version/cache_scope so an entry
            // written by one path is found by the other.
            fn version(&self) -> &'static str {
                "mock-chat-v1"
            }
            fn cache_scope(&self, _model: &str) -> Result<String, ProviderError> {
                Ok("mock-scope".to_string())
            }
        }

        fn state_streaming(store: MockCompletionStore) -> Arc<AppState> {
            Arc::new(state(store))
        }

        async fn drain(body: axum::body::Body) -> String {
            use futures::StreamExt;
            let mut s = body.into_data_stream();
            let mut buf = Vec::new();
            while let Some(c) = s.next().await {
                buf.extend_from_slice(&c.unwrap());
            }
            String::from_utf8(buf).unwrap()
        }

        fn sreq<'a>(
            p: &Arc<MockStreamProvider>,
            owner: [u8; 32],
            body: &'a serde_json::Value,
        ) -> CompletionStreamRequest<'a> {
            CompletionStreamRequest {
                stream_provider: Arc::clone(p)
                    as Arc<dyn zerocache_ports::StreamingChatCompletionProvider>,
                provider_name: "openai",
                api_key: "sk-caller",
                owner_id: owner,
                model: "gpt-4o",
                body,
            }
        }

        #[tokio::test]
        async fn streamed_miss_streams_live_then_a_stream_true_hit_replays_raw_bytes() {
            let frames = vec![
                b"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"}}]}\n\n".to_vec(),
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n".to_vec(),
                b"data: [DONE]\n\n".to_vec(),
            ];
            let provider = Arc::new(MockStreamProvider::ok(frames.clone()));
            let st = state_streaming(MockCompletionStore::empty());
            let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true});

            let out = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            let StreamingOutcome::Stream { body: b, hit, .. } = out else {
                panic!("expected Stream")
            };
            assert!(!hit);
            let streamed = drain(b).await;
            assert!(
                streamed.contains("hel") && streamed.contains("lo") && streamed.contains("[DONE]")
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(provider.calls(), 1);

            let out2 = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            let StreamingOutcome::Stream {
                body: b2,
                hit: hit2,
                hit_kind,
                ..
            } = out2
            else {
                panic!()
            };
            assert!(hit2);
            assert_eq!(hit_kind, Some(HitKind::Exact));
            assert_eq!(provider.calls(), 1, "the hit must not call upstream");
            let replayed = drain(b2).await;
            assert!(replayed.contains("\"content\":\"hel\""));
            assert!(replayed.contains("[DONE]"));
        }

        #[tokio::test]
        async fn a_stream_false_request_hits_an_entry_filled_by_a_streamed_miss() {
            let frames = vec![
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
                b"data: [DONE]\n\n".to_vec(),
            ];
            let provider = Arc::new(MockStreamProvider::ok(frames));
            let st = state_streaming(MockCompletionStore::empty());
            let sbody = json!({"model":"gpt-4o","messages":[{"role":"user","content":"q"}],"temperature":0,"stream":true});
            complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &sbody))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;

            let chat_provider = Arc::new(MockChatProvider::ok(
                json!({"never":"called"}),
                CompletionUsage::default(),
            ));
            let nbody = json!({"model":"gpt-4o","messages":[{"role":"user","content":"q"}],"temperature":0});
            let out = complete(&st, req(&chat_provider, OWNER_A, &nbody))
                .await
                .unwrap();
            assert!(
                out.hit,
                "stream:false must hit the entry a stream:true miss wrote"
            );
            assert_eq!(
                out.response.body["choices"][0]["message"]["content"],
                "answer"
            );
            assert_eq!(chat_provider.call_count(), 0);
        }

        #[tokio::test]
        async fn a_stream_true_request_hits_an_entry_filled_by_a_non_streaming_miss_via_rechunk() {
            let chat_provider = Arc::new(MockChatProvider::ok(
                json!({"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"stored"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}),
                CompletionUsage {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    total_tokens: 4,
                },
            ));
            let st = state_streaming(MockCompletionStore::empty());
            let nbody = json!({"model":"gpt-4o","messages":[{"role":"user","content":"q"}],"temperature":0});
            complete(&st, req(&chat_provider, OWNER_A, &nbody))
                .await
                .unwrap();

            let stream_provider = Arc::new(MockStreamProvider::ok(vec![]));
            let sbody = json!({"model":"gpt-4o","messages":[{"role":"user","content":"q"}],"temperature":0,"stream":true});
            let out = complete_streaming(Arc::clone(&st), sreq(&stream_provider, OWNER_A, &sbody))
                .await
                .unwrap();
            let StreamingOutcome::Stream { body, hit, .. } = out else {
                panic!()
            };
            assert!(hit);
            assert_eq!(stream_provider.calls(), 0);
            let replayed = drain(body).await;
            assert!(replayed.contains("\"content\":\"stored\""));
            assert!(replayed.trim_end().ends_with("data: [DONE]"));
        }

        #[tokio::test]
        async fn an_incomplete_stream_is_not_stored() {
            let frames = vec![
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"
                    .to_vec(),
            ];
            let provider = Arc::new(MockStreamProvider::ok(frames));
            let st = state_streaming(MockCompletionStore::empty());
            let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true});
            let out = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            let StreamingOutcome::Stream { body: b, .. } = out else {
                panic!()
            };
            let _ = drain(b).await;
            tokio::time::sleep(Duration::from_millis(50)).await;

            complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            assert_eq!(
                provider.calls(),
                2,
                "an incomplete stream must not have been cached"
            );
        }

        #[tokio::test]
        async fn an_error_frame_prevents_caching() {
            let frames = vec![
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
                b"data: {\"error\":{\"message\":\"boom\"}}\n\n".to_vec(),
                b"data: [DONE]\n\n".to_vec(),
            ];
            let provider = Arc::new(MockStreamProvider::ok(frames));
            let st = state_streaming(MockCompletionStore::empty());
            let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true});
            let out = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            let StreamingOutcome::Stream { body: b, .. } = out else {
                panic!()
            };
            let _ = drain(b).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            assert_eq!(provider.calls(), 2);
        }

        #[tokio::test]
        async fn a_non_2xx_upstream_stream_returns_an_upstream_error_and_is_not_cached() {
            let provider = Arc::new(MockStreamProvider::status(429, b"slow down".to_vec()));
            let st = state_streaming(MockCompletionStore::empty());
            let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true});
            let out = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            match out {
                StreamingOutcome::UpstreamError(resp) => assert_eq!(resp.status, 429),
                _ => panic!("expected UpstreamError"),
            }
            complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            assert_eq!(provider.calls(), 2);
        }

        #[tokio::test]
        async fn include_usage_is_injected_and_the_usage_chunk_is_withheld_from_a_client_that_did_not_ask(
        ) {
            let frames = vec![
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":1,\"total_tokens\":9}}\n\n".to_vec(),
                b"data: [DONE]\n\n".to_vec(),
            ];
            let provider = Arc::new(MockStreamProvider::ok(frames));
            let st = state_streaming(MockCompletionStore::empty());
            let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true});
            let out = complete_streaming(Arc::clone(&st), sreq(&provider, OWNER_A, &body))
                .await
                .unwrap();
            let StreamingOutcome::Stream { body: b, .. } = out else {
                panic!()
            };
            let streamed = drain(b).await;
            assert!(
                !streamed.contains("\"usage\""),
                "the client did not ask for usage"
            );
            assert!(
                provider.last_request_had_include_usage(),
                "we must have injected it upstream"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;

            let chat_provider =
                Arc::new(MockChatProvider::ok(json!({}), CompletionUsage::default()));
            let nbody = json!({"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0});
            complete(&st, req(&chat_provider, OWNER_A, &nbody))
                .await
                .unwrap();
            let dump = st.metrics.encode();
            assert!(
                dump.contains("zerocache_completion_prompt_tokens_saved_total{provider=\"openai\",stream=\"false\"} 8"),
                "{dump}"
            );
        }
    }

    #[cfg(feature = "semantic")]
    mod semantic {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use zerocache_core::MatchUnit;
        use zerocache_ports::VectorRecord;
        use zerocache_semantic::{SemanticIndex, TextEmbed};

        use super::*;
        use crate::semantic::SemanticState;

        struct ConstEmbedder(AtomicUsize);
        impl TextEmbed for ConstEmbedder {
            fn embed(&self, _t: &str) -> Result<Vec<f32>, zerocache_semantic::SemanticError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                let mut v = vec![0f32; zerocache_semantic::EMBEDDING_DIM];
                v[0] = 1.0;
                Ok(v)
            }
        }

        struct FailEmbedder;
        impl TextEmbed for FailEmbedder {
            fn embed(&self, _t: &str) -> Result<Vec<f32>, zerocache_semantic::SemanticError> {
                Err(zerocache_semantic::SemanticError("boom".into()))
            }
        }

        struct MemVectorStore(Mutex<Vec<VectorRecord>>);
        impl zerocache_ports::CompletionVectorStore for MemVectorStore {
            fn insert(&self, r: VectorRecord) -> Result<(), StoreError> {
                self.0.lock().unwrap().push(r);
                Ok(())
            }
            fn delete(&self, k: &CacheKey, _scope: &[u8; 32]) -> Result<(), StoreError> {
                self.0.lock().unwrap().retain(|r| &r.exact_key != k);
                Ok(())
            }
            fn load_all(&self) -> Result<Vec<VectorRecord>, StoreError> {
                Ok(self.0.lock().unwrap().clone())
            }
            fn changes_since(
                &self,
                _c: Option<String>,
            ) -> Result<zerocache_ports::VectorChanges, StoreError> {
                Ok(zerocache_ports::VectorChanges::default())
            }
        }

        fn state_semantic(
            store: MockCompletionStore,
            embedder: Arc<dyn TextEmbed>,
            threshold: f32,
        ) -> AppState {
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
                completion_in_flight: Mutex::new(HashMap::new()),
                coordinator: Arc::new(crate::coalesce::NoopCoordinator),
                semantic: Some(SemanticState {
                    embedder,
                    index: Arc::new(SemanticIndex::new()),
                    vector_store: Arc::new(MemVectorStore(Mutex::new(Vec::new()))),
                    threshold,
                    match_unit: MatchUnit::LastUser,
                }),
            }
        }

        fn body(user: &str, max_tokens: u32) -> serde_json::Value {
            json!({"model":"gpt-4o","messages":[
                {"role":"system","content":"support bot"},
                {"role":"user","content": user}
            ],"temperature":0,"max_tokens": max_tokens})
        }

        fn sreq<'a>(
            p: &Arc<MockChatProvider>,
            owner: [u8; 32],
            body: &'a serde_json::Value,
        ) -> CompletionRequest<'a> {
            CompletionRequest {
                provider: Arc::clone(p) as Arc<dyn ChatCompletionProvider>,
                provider_name: "openai",
                api_key: "sk-caller",
                owner_id: owner,
                model: "gpt-4o",
                body,
            }
        }

        #[tokio::test]
        async fn a_paraphrase_with_the_same_coarse_key_is_a_semantic_hit() {
            let provider = Arc::new(MockChatProvider::ok(
                json!({"choices":[{"message":{"content":"A"}}]}),
                CompletionUsage {
                    prompt_tokens: 30,
                    completion_tokens: 6,
                    total_tokens: 36,
                },
            ));
            let emb = Arc::new(ConstEmbedder(AtomicUsize::new(0)));
            let st = state_semantic(MockCompletionStore::empty(), emb, 0.97);

            let first = body("how do I reset my password?", 256);
            let o1 = complete(&st, sreq(&provider, OWNER_A, &first))
                .await
                .unwrap();
            assert!(!o1.hit);
            assert_eq!(provider.call_count(), 1);

            let second = body("how can i reset the password", 256);
            let o2 = complete(&st, sreq(&provider, OWNER_A, &second))
                .await
                .unwrap();
            assert!(o2.hit, "a paraphrase must be a semantic hit");
            assert_eq!(o2.hit_kind, Some(HitKind::Semantic));
            assert!(o2.semantic_score.unwrap() > 0.99);
            assert_eq!(provider.call_count(), 1);
            assert_eq!(
                o2.response.body,
                json!({"choices":[{"message":{"content":"A"}}]})
            );

            let dump = st.metrics.encode();
            assert!(
                dump.contains(
                    "zerocache_completion_semantic_hits_total{provider=\"openai\",stream=\"false\"} 1"
                ),
                "{dump}"
            );
            assert!(
                dump.contains(
                    "zerocache_completion_prompt_tokens_saved_total{provider=\"openai\",stream=\"false\"} 30"
                ),
                "{dump}"
            );
        }

        #[tokio::test]
        async fn an_exact_hit_never_touches_the_embedder() {
            let provider = Arc::new(MockChatProvider::ok(
                json!({"x": 1}),
                CompletionUsage::default(),
            ));
            let emb = Arc::new(ConstEmbedder(AtomicUsize::new(0)));
            let calls = emb.clone();
            let st = state_semantic(MockCompletionStore::empty(), emb, 0.97);
            let b = body("identical", 256);
            complete(&st, sreq(&provider, OWNER_A, &b)).await.unwrap();
            let after_miss = calls.0.load(Ordering::SeqCst);
            complete(&st, sreq(&provider, OWNER_A, &b)).await.unwrap();
            assert_eq!(calls.0.load(Ordering::SeqCst), after_miss);
        }

        #[tokio::test]
        async fn a_different_coarse_key_is_not_a_semantic_hit() {
            let provider = Arc::new(MockChatProvider::ok(
                json!({"x": 1}),
                CompletionUsage::default(),
            ));
            let emb = Arc::new(ConstEmbedder(AtomicUsize::new(0)));
            let st = state_semantic(MockCompletionStore::empty(), emb, 0.97);
            complete(&st, sreq(&provider, OWNER_A, &body("q one", 256)))
                .await
                .unwrap();
            let o = complete(&st, sreq(&provider, OWNER_A, &body("q two", 512)))
                .await
                .unwrap();
            assert!(!o.hit);
            assert_eq!(provider.call_count(), 2);
        }

        #[tokio::test]
        async fn an_embed_failure_falls_back_to_the_provider() {
            let provider = Arc::new(MockChatProvider::ok(
                json!({"x": 1}),
                CompletionUsage::default(),
            ));
            let st = state_semantic(MockCompletionStore::empty(), Arc::new(FailEmbedder), 0.97);
            complete(&st, sreq(&provider, OWNER_A, &body("a", 256)))
                .await
                .unwrap();
            let o = complete(&st, sreq(&provider, OWNER_A, &body("b", 256)))
                .await
                .unwrap();
            assert!(!o.hit);
            assert_eq!(provider.call_count(), 2);
        }

        #[tokio::test]
        async fn below_threshold_is_not_a_hit() {
            let provider = Arc::new(MockChatProvider::ok(
                json!({"x": 1}),
                CompletionUsage::default(),
            ));
            let emb = Arc::new(ConstEmbedder(AtomicUsize::new(0)));
            let st = state_semantic(MockCompletionStore::empty(), emb, 1.0001);
            complete(&st, sreq(&provider, OWNER_A, &body("a", 256)))
                .await
                .unwrap();
            let o = complete(&st, sreq(&provider, OWNER_A, &body("b", 256)))
                .await
                .unwrap();
            assert!(!o.hit);
            assert_eq!(provider.call_count(), 2);
        }
    }
}

mod app;
mod coalesce;
mod completion;
mod config;
mod dashboard;
mod image;
mod otel;
#[cfg(feature = "semantic")]
mod semantic;
mod wire;

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use tower_http::trace::TraceLayer;

use app::{
    check_store_readiness, delete_batch, embed_batch, AppError, AppState, DeleteRequest,
    EmbedRequest, Metrics,
};
use completion::{complete, CompletionRequest};
use config::{
    Config, StorageBackend, DEFAULT_GEMINI_BASE_URL, DEFAULT_HUGGINGFACE_BASE_URL,
    DEFAULT_MISTRAL_BASE_URL, DEFAULT_OPENAI_BASE_URL,
};
use wire::{
    DeleteResponse, EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse, Usage,
};
use zerocache_adapters_azure::new_provider as new_azure_provider;
use zerocache_adapters_bedrock::new_provider as new_bedrock_provider;
use zerocache_adapters_gemini::GeminiProvider;
use zerocache_adapters_huggingface::HuggingFaceProvider;
use zerocache_adapters_mistral::MistralProvider;
use zerocache_adapters_openai::{OpenAiProvider, OpenAiWireChatProvider};
use zerocache_adapters_redis::RedisStore;
use zerocache_adapters_sled::SledStore;
use zerocache_adapters_vertexai::new_provider as new_vertexai_provider;
use zerocache_core::derive_owner_id;
use zerocache_ports::{CompletionStore, EmbeddingProvider, EmbeddingStore, ImageEmbeddingProvider};

#[tokio::main]
async fn main() {
    // `zerocache-http --health-check` probes the local /health endpoint and
    // exits 0/1. This is the container HEALTHCHECK -- the `FROM scratch` image
    // has no curl/wget/shell, so the binary is its own probe.
    if std::env::args().any(|a| a == "--health-check") {
        std::process::exit(health_check_probe());
    }

    let tracer_provider = otel::init();
    let config = Config::from_env();
    log_overridden_base_urls(&config);
    log_chat_providers(&config);

    #[cfg(not(feature = "semantic"))]
    if config.semantic_enabled {
        tracing::info!(
            "ZEROCACHE_SEMANTIC is set but this binary was built without the `semantic` feature -- ignoring"
        );
    }

    // One concrete store instance, exposed as multiple trait objects: the
    // embedding, completion, and (sled only) vector-store paths all share the
    // same sled DB / redis pool -- opening a second handle to the same sled
    // directory would fail its exclusive lock.
    #[cfg(feature = "semantic")]
    let completion_vector_store: Option<Arc<dyn zerocache_ports::CompletionVectorStore>>;
    let (store, completion_store): (Arc<dyn EmbeddingStore>, Arc<dyn CompletionStore>) =
        match config.storage_backend {
            StorageBackend::Sled => {
                let sled = Arc::new(
                    SledStore::open(&config.storage_path, config.ttl)
                        .expect("failed to open sled store"),
                );
                #[cfg(feature = "semantic")]
                {
                    completion_vector_store =
                        Some(Arc::clone(&sled) as Arc<dyn zerocache_ports::CompletionVectorStore>);
                }
                (
                    Arc::clone(&sled) as Arc<dyn EmbeddingStore>,
                    sled as Arc<dyn CompletionStore>,
                )
            }
            StorageBackend::Redis => {
                let redis = RedisStore::connect(&config.redis_url, config.ttl)
                    .expect("failed to connect to redis");
                #[cfg(feature = "semantic")]
                let redis = redis.with_semantic_index_maxlen(config.semantic_index_maxlen);
                let redis = Arc::new(redis);
                #[cfg(feature = "semantic")]
                {
                    completion_vector_store =
                        Some(Arc::clone(&redis) as Arc<dyn zerocache_ports::CompletionVectorStore>);
                }
                (
                    Arc::clone(&redis) as Arc<dyn EmbeddingStore>,
                    redis as Arc<dyn CompletionStore>,
                )
            }
        };

    let gemini_provider = Arc::new(GeminiProvider::new(config.gemini_base_url.clone()));

    let mut providers: HashMap<String, Arc<dyn EmbeddingProvider>> = HashMap::new();
    providers.insert(
        "openai".to_string(),
        Arc::new(OpenAiProvider::new(config.openai_base_url.clone())),
    );
    providers.insert(
        "mistral".to_string(),
        Arc::new(MistralProvider::new(config.mistral_base_url.clone())),
    );
    providers.insert(
        "gemini".to_string(),
        Arc::clone(&gemini_provider) as Arc<dyn EmbeddingProvider>,
    );
    providers.insert(
        "huggingface".to_string(),
        Arc::new(HuggingFaceProvider::new(
            config.huggingface_base_url.clone(),
        )),
    );

    let mut image_providers: HashMap<String, Arc<dyn ImageEmbeddingProvider>> = HashMap::new();
    image_providers.insert(
        "gemini".to_string(),
        gemini_provider as Arc<dyn ImageEmbeddingProvider>,
    );

    // Registered unconditionally: both are constructible from pure defaults,
    // and a missing per-request coordinate (Vertex project, Bedrock region)
    // is a per-request resolution error, not a startup one.
    providers.insert(
        "bedrock".to_string(),
        Arc::new(new_bedrock_provider(
            config.bedrock_region.clone(),
            config.bedrock_endpoint_template.clone(),
        )),
    );
    providers.insert(
        "vertexai".to_string(),
        Arc::new(new_vertexai_provider(
            config.vertex_project.clone(),
            config.vertex_location.clone(),
            config.vertex_endpoint_template.clone(),
        )),
    );

    // Azure is the exception: an Azure resource name *is* its hostname, so
    // there is no default endpoint to fall back to. Either surface alone is
    // enough to register the provider -- a Foundry-only deployment (no Azure
    // OpenAI resource at all) is a real, supported configuration, and
    // AzureRouter::resolve returns a clean ProviderError naming the missing
    // env var if a caller's model then targets the unconfigured surface.
    // With both env vars unset, POST /azure/v1/embeddings returns the
    // existing 404 "unknown provider" straight out of the missing map key --
    // the same structural mechanism that already makes
    // /openai/v1/images/embeddings 404, with no extra enforcement code.
    if config.azure_openai_base_url.is_some() || config.azure_foundry_base_url.is_some() {
        providers.insert(
            "azure".to_string(),
            Arc::new(new_azure_provider(
                config.azure_openai_base_url.clone(),
                config.azure_foundry_base_url.clone(),
                config.azure_foundry_api_version.clone(),
                config.azure_auth_mode,
            )),
        );
    } else {
        tracing::info!(
            "neither ZEROCACHE_AZURE_OPENAI_BASE_URL nor ZEROCACHE_AZURE_FOUNDRY_BASE_URL is set -- the 'azure' provider is not registered and /azure/v1/embeddings will return 404"
        );
    }

    let port = config.port;

    // Chat-completion providers, keyed by `{provider}` path segment: one
    // OpenAiWireChatProvider per entry in the merged chat-provider registry
    // (BUILTIN_CHAT_PROVIDERS + ZEROCACHE_CHAT_PROVIDERS). An unregistered
    // name 404s out of the missing key.
    let mut completion_providers: HashMap<
        String,
        Arc<dyn zerocache_ports::ChatCompletionProvider>,
    > = HashMap::new();
    for (name, url) in &config.chat_providers {
        completion_providers.insert(
            name.clone(),
            Arc::new(OpenAiWireChatProvider::new(url.clone())),
        );
    }

    #[cfg(feature = "semantic")]
    let (semantic, semantic_poll_cursor) = match completion_vector_store {
        Some(vs) => match semantic::build_semantic_state(&config, vs) {
            Some(st) => {
                // rebuild_index runs a blocking paged XRANGE inline; intentional here —
                // this is before axum::serve, nothing else is scheduled on the runtime yet.
                // Do not move it after the server starts without wrapping it in spawn_blocking.
                let cursor = semantic::rebuild_index(&st, &config);
                (Some(st), cursor)
            }
            None => (None, None),
        },
        None => (None, None),
    };

    let state = Arc::new(AppState {
        store,
        providers,
        image_providers,
        metrics: Metrics::new(),
        in_flight: std::sync::Mutex::new(HashMap::new()),
        image_in_flight: std::sync::Mutex::new(HashMap::new()),
        completion_store,
        completion_providers,
        completion_in_flight: std::sync::Mutex::new(HashMap::new()),
        coordinator: Arc::new(coalesce::NoopCoordinator),
        #[cfg(feature = "semantic")]
        semantic,
    });

    #[cfg(feature = "semantic")]
    if let Some(cursor) = semantic_poll_cursor {
        tokio::spawn(semantic::run_semantic_poll_task(
            Arc::clone(&state),
            cursor,
            std::time::Duration::from_millis(config.semantic_poll_ms),
        ));
    }

    let app = Router::new()
        .route(
            "/:provider/v1/embeddings",
            post(embeddings_handler).delete(delete_handler),
        )
        .route(
            "/:provider/v1/images/embeddings",
            post(image_embeddings_handler).delete(delete_image_handler),
        )
        .route(
            "/:provider/v1/chat/completions",
            post(chat_completions_handler),
        )
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/dashboard", get(dashboard::index))
        .route("/dashboard/", get(dashboard::index))
        .route("/dashboard/*path", get(dashboard::asset))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind port");
    tracing::info!(port, "zerocache-http listening");
    tracing::info!("savings dashboard at http://localhost:{port}/dashboard");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    // The batch span exporter buffers spans in memory; an unflushed exit
    // can silently drop whatever hasn't been sent yet. Only relevant when
    // OTLP export is actually enabled (otel::init returns None otherwise).
    if let Some(provider) = tracer_provider {
        if let Err(e) = provider.shutdown() {
            eprintln!("warning: failed to flush OpenTelemetry spans on shutdown: {e}");
        }
    }
}

/// Notes at startup when an operator has repointed a provider's base URL away
/// from its real default -- e.g. at a self-hosted vLLM/LM Studio/TGI instance.
///
/// This used to be a `warn!` about stale cache hits: `CacheKey::derive` did
/// not include the endpoint, so repointing an adapter silently reused vectors
/// computed by the previous upstream. That hazard is gone -- the cache key now
/// carries `EmbeddingProvider::cache_scope`, which for these four adapters is
/// the configured base URL, so repointing produces a clean cold cache instead
/// of a wrong hit. Kept at `info` because "your hit rate is about to drop to
/// zero" is still worth saying out loud once at startup; it just is not a
/// correctness warning anymore.
fn log_overridden_base_urls(config: &Config) {
    let overrides = [
        (
            "ZEROCACHE_OPENAI_BASE_URL",
            &config.openai_base_url,
            DEFAULT_OPENAI_BASE_URL,
        ),
        (
            "ZEROCACHE_MISTRAL_BASE_URL",
            &config.mistral_base_url,
            DEFAULT_MISTRAL_BASE_URL,
        ),
        (
            "ZEROCACHE_GEMINI_BASE_URL",
            &config.gemini_base_url,
            DEFAULT_GEMINI_BASE_URL,
        ),
        (
            "ZEROCACHE_HUGGINGFACE_BASE_URL",
            &config.huggingface_base_url,
            DEFAULT_HUGGINGFACE_BASE_URL,
        ),
    ];

    for (env_var_name, actual, default) in overrides {
        if actual != default {
            tracing::info!(
                "{env_var_name} is overridden to '{actual}' -- cache entries are keyed by endpoint, so requests against this endpoint start from a cold cache rather than reusing anything cached under the default endpoint"
            );
        }
    }
}

/// Logs the resolved chat-provider registry so a typo in
/// ZEROCACHE_CHAT_PROVIDERS shows up at boot, not on first request. Chat
/// provider URLs carry no secrets (BYOK: the key is per-request).
fn log_chat_providers(config: &Config) {
    for (name, url) in &config.chat_providers {
        tracing::info!("chat provider '{name}' -> {url}/chat/completions");
    }
}

/// Resolves on Ctrl+C (works on every platform, including the Windows dev
/// environment) or, on Unix only, SIGTERM (what Kubernetes sends a pod
/// before force-killing it after the grace period). Whichever fires first
/// wins; the other branch is simply dropped.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }

    tracing::info!("shutdown signal received, finishing in-flight requests");
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    value.strip_prefix("Bearer ").map(|s| s.to_string())
}

/// Axum's default `Json<T>` extractor rejection is plain text, not this
/// app's `{"error": "..."}` shape every other error path uses -- a real gap
/// found via LangChain TS battle-testing (a consumer parsing all error
/// bodies uniformly broke on this one path). `.status()`/`.body_text()` are
/// generated by axum-core's `composite_rejection!` macro for every
/// `JsonRejection` variant, so this preserves the exact same status code
/// (400 for invalid JSON syntax, 422 for valid JSON that doesn't match the
/// target shape, etc.) and message axum already computed correctly --
/// only the response envelope changes.
fn json_rejection_to_error_response(rejection: JsonRejection) -> (StatusCode, Json<ErrorResponse>) {
    (
        rejection.status(),
        Json(ErrorResponse {
            error: rejection.body_text(),
        }),
    )
}

#[tracing::instrument(skip_all, fields(provider = %provider_name))]
async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let Json(request) = body.map_err(json_rejection_to_error_response)?;

    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')"
                    .to_string(),
            }),
        )
    })?;

    let provider = state
        .providers
        .get(&provider_name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("unknown provider '{provider_name}'"),
                }),
            )
        })?;

    let owner_id = derive_owner_id(&api_key);
    let model = request.model;
    let texts = request.input;

    let embed_request = EmbedRequest {
        provider,
        provider_name: &provider_name,
        api_key: &api_key,
        owner_id,
        model: &model,
        texts: &texts,
    };

    let result = embed_batch(&state, embed_request).await;

    let (vectors, stats) = result.map_err(|err| {
        let status = match &err {
            AppError::Provider(_) => StatusCode::BAD_GATEWAY,
            AppError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            object: "embedding",
            embedding,
            index,
        })
        .collect();

    let mut response = Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: Usage {
            prompt_tokens: stats.provider_prompt_tokens,
            total_tokens: stats.provider_total_tokens,
        },
    })
    .into_response();

    let headers = response.headers_mut();
    headers.insert(
        "x-zerocache-hits",
        stats
            .hits
            .to_string()
            .parse()
            .expect("digit string is a valid header value"),
    );
    headers.insert(
        "x-zerocache-misses",
        stats
            .misses
            .to_string()
            .parse()
            .expect("digit string is a valid header value"),
    );

    Ok(response)
}

#[tracing::instrument(skip_all, fields(provider = %provider_name))]
async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let Json(request) = body.map_err(json_rejection_to_error_response)?;

    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')"
                    .to_string(),
            }),
        )
    })?;

    let provider = state
        .providers
        .get(&provider_name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("unknown provider '{provider_name}'"),
                }),
            )
        })?;

    let owner_id = derive_owner_id(&api_key);
    let model = request.model;
    let texts = request.input;

    let delete_request = DeleteRequest {
        provider: provider.as_ref(),
        provider_name: &provider_name,
        owner_id,
        model: &model,
        texts: &texts,
    };

    let deleted = delete_batch(&state, delete_request).await.map_err(|err| {
        let status = match &err {
            AppError::Provider(_) => StatusCode::BAD_GATEWAY,
            AppError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })?;

    Ok(Json(DeleteResponse { deleted }))
}

#[tracing::instrument(skip_all, fields(provider = %provider_name))]
async fn image_embeddings_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let Json(request) = body.map_err(json_rejection_to_error_response)?;

    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')"
                    .to_string(),
            }),
        )
    })?;

    let provider = state
        .image_providers
        .get(&provider_name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("provider '{provider_name}' does not support image embeddings"),
                }),
            )
        })?;

    let owner_id = derive_owner_id(&api_key);
    let model = request.model;

    let images: Vec<zerocache_ports::ImageInput> = request
        .input
        .iter()
        .map(|uri| image::parse_data_uri(uri))
        .collect::<Result<_, _>>()
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    let embed_request = app::EmbedImageRequest {
        provider,
        provider_name: &provider_name,
        api_key: &api_key,
        owner_id,
        model: &model,
        images: &images,
    };

    let result = app::embed_image_batch(&state, embed_request).await;

    let (vectors, stats) = result.map_err(|err| {
        let status = match &err {
            AppError::Provider(_) => StatusCode::BAD_GATEWAY,
            AppError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            object: "embedding",
            embedding,
            index,
        })
        .collect();

    let mut response = Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: Usage {
            prompt_tokens: stats.provider_prompt_tokens,
            total_tokens: stats.provider_total_tokens,
        },
    })
    .into_response();

    let headers = response.headers_mut();
    headers.insert(
        "x-zerocache-hits",
        stats
            .hits
            .to_string()
            .parse()
            .expect("digit string is a valid header value"),
    );
    headers.insert(
        "x-zerocache-misses",
        stats
            .misses
            .to_string()
            .parse()
            .expect("digit string is a valid header value"),
    );

    Ok(response)
}

#[tracing::instrument(skip_all, fields(provider = %provider_name))]
async fn delete_image_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let Json(request) = body.map_err(json_rejection_to_error_response)?;

    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')"
                    .to_string(),
            }),
        )
    })?;

    let provider = state
        .image_providers
        .get(&provider_name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("provider '{provider_name}' does not support image embeddings"),
                }),
            )
        })?;

    let owner_id = derive_owner_id(&api_key);
    let model = request.model;

    let images: Vec<zerocache_ports::ImageInput> = request
        .input
        .iter()
        .map(|uri| image::parse_data_uri(uri))
        .collect::<Result<_, _>>()
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;

    let delete_request = app::DeleteImageRequest {
        provider: provider.as_ref(),
        provider_name: &provider_name,
        owner_id,
        model: &model,
        images: &images,
    };

    let deleted = app::delete_image_batch(&state, delete_request)
        .await
        .map_err(|err| {
            let status = match &err {
                AppError::Provider(_) => StatusCode::BAD_GATEWAY,
                AppError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            )
        })?;

    Ok(Json(DeleteResponse { deleted }))
}

/// `POST /{provider}/v1/chat/completions` -- the semantic completion cache.
/// The request/response body is the OpenAI chat shape, forwarded verbatim
/// to the upstream provider on a miss; a hit replays the stored body with a
/// `200`. An `X-Zerocache-Completion-Hit: true|false` header says which.
///
/// Only deterministic requests (temperature 0 or an explicit seed) are ever
/// cached -- anything else is a transparent passthrough. See
/// `crate::completion`.
#[tracing::instrument(skip_all, fields(provider = %provider_name))]
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let Json(request_body) = body.map_err(json_rejection_to_error_response)?;

    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')"
                    .to_string(),
            }),
        )
    })?;

    let provider = state
        .completion_providers
        .get(&provider_name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("unknown provider '{provider_name}'"),
                }),
            )
        })?;

    let model = request_body
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: "request body is missing a string 'model' field".to_string(),
                }),
            )
        })?
        .to_string();

    let owner_id = derive_owner_id(&api_key);

    let outcome = complete(
        &state,
        CompletionRequest {
            provider,
            provider_name: &provider_name,
            api_key: &api_key,
            owner_id,
            model: &model,
            body: &request_body,
        },
    )
    .await
    .map_err(|err| {
        let status = match &err {
            AppError::Provider(_) => StatusCode::BAD_GATEWAY,
            AppError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })?;

    // Forward the upstream status verbatim (a hit is always 200); a value
    // outside the valid range would only come from a broken upstream, so
    // fall back to 502 rather than panic.
    let status = StatusCode::from_u16(outcome.response.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = (status, Json(outcome.response.body)).into_response();
    response.headers_mut().insert(
        "x-zerocache-completion-hit",
        if outcome.hit { "true" } else { "false" }
            .parse()
            .expect("static ascii is a valid header value"),
    );
    if let Some(kind) = outcome.hit_kind {
        response.headers_mut().insert(
            "x-zerocache-completion-hit-kind",
            match kind {
                completion::HitKind::Exact => "exact",
                completion::HitKind::Semantic => "semantic",
            }
            .parse()
            .expect("static ascii is a valid header value"),
        );
    }
    if let Some(score) = outcome.semantic_score {
        if let Ok(v) = format!("{score:.3}").parse() {
            response
                .headers_mut()
                .insert("x-zerocache-semantic-score", v);
        }
    }
    Ok(response)
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        state.metrics.encode(),
    )
}

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

/// Blocking one-shot HTTP/1.0 GET to the local `/health` endpoint. Returns a
/// process exit code: 0 if the server answered `200`, 1 otherwise. Uses only
/// `std::net` so it adds no dependency and works in the dependency-free
/// `FROM scratch` image.
fn health_check_probe() -> i32 {
    use std::io::{Read, Write};

    let port: u16 = std::env::var("ZEROCACHE_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);
    let timeout = std::time::Duration::from_secs(3);

    let mut stream = match std::net::TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));

    if stream
        .write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return 1;
    }

    let mut buf = String::new();
    if stream.read_to_string(&mut buf).is_err() {
        return 1;
    }

    let ok = buf.starts_with("HTTP/1.0 200") || buf.starts_with("HTTP/1.1 200");
    i32::from(!ok)
}

async fn ready_handler(State(state): State<Arc<AppState>>) -> StatusCode {
    match check_store_readiness(&state).await {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

mod app;
mod config;
mod wire;

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

use app::{delete_batch, embed_batch, AppError, AppState, DeleteRequest, EmbedRequest, Metrics};
use config::{Config, StorageBackend};
use wire::{DeleteResponse, EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse, Usage};
use zerocache_adapters_gemini::GeminiProvider;
use zerocache_adapters_mistral::MistralProvider;
use zerocache_adapters_openai::OpenAiProvider;
use zerocache_adapters_redis::RedisStore;
use zerocache_adapters_sled::SledStore;
use zerocache_core::derive_owner_id;
use zerocache_ports::{EmbeddingProvider, EmbeddingStore};

#[tokio::main]
async fn main() {
    let config = Config::from_env();

    let store: Arc<dyn EmbeddingStore> = match config.storage_backend {
        StorageBackend::Sled => {
            Arc::new(SledStore::open(&config.storage_path, config.ttl).expect("failed to open sled store"))
        }
        StorageBackend::Redis => {
            Arc::new(RedisStore::connect(&config.redis_url, config.ttl).expect("failed to connect to redis"))
        }
    };

    let mut providers: HashMap<String, Arc<dyn EmbeddingProvider>> = HashMap::new();
    providers.insert("openai".to_string(), Arc::new(OpenAiProvider::new("https://api.openai.com")));
    providers.insert("mistral".to_string(), Arc::new(MistralProvider::new("https://api.mistral.ai")));
    providers.insert(
        "gemini".to_string(),
        Arc::new(GeminiProvider::new("https://generativelanguage.googleapis.com")),
    );

    let port = config.port;

    let state = Arc::new(AppState {
        store,
        providers,
        metrics: Metrics::new(),
    });

    let app = Router::new()
        .route(
            "/:provider/v1/embeddings",
            post(embeddings_handler).delete(delete_handler),
        )
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind port");
    println!("zerocache-http listening on 0.0.0.0:{port}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
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

    println!("shutdown signal received, finishing in-flight requests");
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(axum::http::header::AUTHORIZATION)?.to_str().ok()?;
    value.strip_prefix("Bearer ").map(|s| s.to_string())
}

async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')".to_string(),
            }),
        )
    })?;

    let provider = state.providers.get(&provider_name).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("unknown provider '{provider_name}'") }),
        )
    })?;

    let owner_id = derive_owner_id(&api_key);
    let model = request.model;
    let texts = request.input;

    let embed_request = EmbedRequest {
        provider: provider.as_ref(),
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
        (status, Json(ErrorResponse { error: err.to_string() }))
    })?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject { embedding, index })
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
        stats.hits.to_string().parse().expect("digit string is a valid header value"),
    );
    headers.insert(
        "x-zerocache-misses",
        stats.misses.to_string().parse().expect("digit string is a valid header value"),
    );

    Ok(response)
}

async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Path(provider_name): Path<String>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let api_key = extract_bearer_token(&headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "missing or malformed Authorization header (expected 'Bearer <key>')".to_string(),
            }),
        )
    })?;

    let provider = state.providers.get(&provider_name).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("unknown provider '{provider_name}'") }),
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
        (status, Json(ErrorResponse { error: err.to_string() }))
    })?;

    Ok(Json(DeleteResponse { deleted }))
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.encode(),
    )
}

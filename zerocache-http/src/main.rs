mod app;
mod config;
mod wire;

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

use app::{embed_batch, AppError, AppState, Metrics};
use config::{Config, StorageBackend};
use wire::{EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse, Usage};
use zerocache_adapters_openai::OpenAiProvider;
use zerocache_adapters_redis::RedisStore;
use zerocache_adapters_sled::SledStore;
use zerocache_ports::EmbeddingStore;

// Not `#[tokio::main]`: `OpenAiProvider` holds a `reqwest::blocking::Client`,
// which builds its own internal Tokio runtime at construction time. Building
// (or later dropping) that internal runtime while already inside another
// runtime's context panics — so the provider must be constructed before any
// Tokio runtime exists, and the runtime entered explicitly afterward.
fn main() {
    let config = Config::from_env();

    let store: Arc<dyn EmbeddingStore> = match config.storage_backend {
        StorageBackend::Sled => {
            Arc::new(SledStore::open(&config.storage_path).expect("failed to open sled store"))
        }
        StorageBackend::Redis => {
            Arc::new(RedisStore::connect(&config.redis_url).expect("failed to connect to redis"))
        }
    };
    let provider = OpenAiProvider::new(&config.provider_base_url, &config.provider_api_key);
    let port = config.port;

    let state = Arc::new(AppState {
        store,
        provider: Arc::new(provider),
        model_version: "v1".to_string(),
        metrics: Metrics::new(),
    });

    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async move {
        let app = Router::new()
            .route("/v1/embeddings", post(embeddings_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
            .await
            .expect("failed to bind port");
        println!("zerocache-http listening on 0.0.0.0:{port}");
        axum::serve(listener, app).await.expect("server error");
    });
}

async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let model = request.model;
    let texts = request.input;
    let model_for_provider = model.clone();

    let result = tokio::task::spawn_blocking(move || embed_batch(&state, &model_for_provider, &texts))
        .await
        .expect("embed_batch task panicked");

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

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.encode(),
    )
}

mod app;
mod config;
mod wire;

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};

use app::{embed_batch, AppError, AppState};
use config::{Config, StorageBackend};
use wire::{EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, ErrorResponse, Usage};
use zerocache_adapters_openai::OpenAiProvider;
use zerocache_adapters_redis::RedisStore;
use zerocache_adapters_sled::SledStore;
use zerocache_ports::EmbeddingStore;

#[tokio::main]
async fn main() {
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

    let state = Arc::new(AppState {
        store,
        provider: Arc::new(provider),
        model_version: "v1".to_string(),
    });

    let app = Router::new()
        .route("/v1/embeddings", post(embeddings_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port))
        .await
        .expect("failed to bind port");
    axum::serve(listener, app).await.expect("server error");
}

async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model = request.model;
    let texts = request.input;
    let model_for_provider = model.clone();

    let vectors = tokio::task::spawn_blocking(move || embed_batch(&state, &model_for_provider, &texts))
        .await
        .expect("embed_batch task panicked");

    let vectors = vectors.map_err(|err| {
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

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: Usage { prompt_tokens: 0, total_tokens: 0 },
    }))
}

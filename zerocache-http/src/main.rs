mod app;
mod config;
mod wire;

use std::sync::Arc;

use axum::{extract::State, routing::post, Json, Router};

use app::{embed_batch, AppState};
use config::Config;
use wire::{EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, Usage};
use zerocache_adapters_openai::OpenAiProvider;
use zerocache_adapters_sled::SledStore;

#[tokio::main]
async fn main() {
    let config = Config::from_env();

    let store = SledStore::open(&config.storage_path).expect("failed to open sled store");
    let provider = OpenAiProvider::new(&config.provider_base_url, &config.provider_api_key);

    let state = Arc::new(AppState {
        store: Arc::new(store),
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
) -> Json<EmbeddingsResponse> {
    let model = request.model;
    let texts = request.input;
    let model_for_provider = model.clone();

    let vectors = tokio::task::spawn_blocking(move || embed_batch(&state, &model_for_provider, &texts))
        .await
        .expect("embed_batch panicked");

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject { embedding, index })
        .collect();

    Json(EmbeddingsResponse {
        object: "list",
        data,
        model,
        usage: Usage { prompt_tokens: 0, total_tokens: 0 },
    })
}

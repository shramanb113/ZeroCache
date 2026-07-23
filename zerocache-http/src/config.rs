pub enum StorageBackend {
    // Embedded, single-process. Fine for local dev or a single-replica
    // deployment; cannot be shared across multiple Kubernetes pods.
    Sled,
    // Shared, network-accessible. Required for multi-replica deployments so
    // every pod hits the same cache instead of each keeping a private one.
    Redis,
}

pub struct Config {
    pub port: u16,
    pub storage_backend: StorageBackend,
    pub storage_path: String,
    pub redis_url: String,
    pub provider_base_url: String,
    pub provider_api_key: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("ZEROCACHE_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            storage_backend: match std::env::var("ZEROCACHE_STORAGE_BACKEND").as_deref() {
                Ok("redis") => StorageBackend::Redis,
                _ => StorageBackend::Sled,
            },
            storage_path: std::env::var("ZEROCACHE_STORAGE_PATH").unwrap_or_else(|_| "./data".into()),
            redis_url: std::env::var("ZEROCACHE_REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            provider_base_url: std::env::var("ZEROCACHE_PROVIDER_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".into()),
            provider_api_key: std::env::var("ZEROCACHE_PROVIDER_API_KEY")
                .expect("ZEROCACHE_PROVIDER_API_KEY must be set"),
        }
    }
}

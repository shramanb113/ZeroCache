pub struct Config {
    pub port: u16,
    pub storage_path: String,
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
            storage_path: std::env::var("ZEROCACHE_STORAGE_PATH").unwrap_or_else(|_| "./data".into()),
            provider_base_url: std::env::var("ZEROCACHE_PROVIDER_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".into()),
            provider_api_key: std::env::var("ZEROCACHE_PROVIDER_API_KEY")
                .expect("ZEROCACHE_PROVIDER_API_KEY must be set"),
        }
    }
}

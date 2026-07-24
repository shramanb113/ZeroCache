//! OpenTelemetry tracing setup.
//!
//! Entirely optional: if `OTEL_EXPORTER_OTLP_ENDPOINT` isn't set, the server
//! still logs via a plain console `tracing_subscriber::fmt` layer -- no OTLP
//! collector is required to run Zerocache locally or in any deployment that
//! doesn't want distributed tracing. Matches the existing pattern for other
//! optional, env-var-gated behavior (e.g. `ZEROCACHE_TTL_SECONDS`).
//!
//! Uses OTLP over gRPC (`grpc-tonic`), not the HTTP exporter: the HTTP
//! exporter's default feature set pulls in `reqwest 0.13` via
//! `reqwest-blocking-client`, which conflicts with the `reqwest 0.12`
//! already pinned workspace-wide across the three provider adapters (same
//! conflict class hit and worked around for `reqwest-retry`/
//! `reqwest-middleware`). gRPC/tonic has no such dependency and is also the
//! more common default transport most OTLP collectors expect on port 4317.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Initializes the global `tracing` subscriber. Returns the OTel tracer
/// provider when OTLP export is enabled, so the caller can flush it before
/// the process exits -- the batch exporter buffers spans in memory, and an
/// unflushed shutdown can silently drop the last batch.
pub fn init() -> Option<SdkTracerProvider> {
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    let tracer_provider = otlp_endpoint.as_ref().map(|endpoint| {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .build()
            .unwrap_or_else(|e| {
                panic!(
                    "OTEL_EXPORTER_OTLP_ENDPOINT='{endpoint}' is set but the OTLP exporter \
                     could not be configured: {e} -- check the endpoint is a valid gRPC URL \
                     (e.g. http://localhost:4317)"
                )
            });
        let resource = Resource::builder().with_service_name("zerocache-http").build();
        SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build()
    });

    let otel_layer = tracer_provider
        .as_ref()
        .map(|provider| tracing_opentelemetry::layer().with_tracer(provider.tracer("zerocache-http")));

    // Without an explicit filter, tracing_subscriber's default is "enable
    // everything" -- every TRACE-level event from every dependency (sled's
    // internals are especially chatty). RUST_LOG still overrides this, same
    // as any other tracing-subscriber-based binary.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    if let Some(endpoint) = &otlp_endpoint {
        tracing::info!(endpoint = %endpoint, "OpenTelemetry tracing enabled");
    }

    tracer_provider
}

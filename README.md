# Zerocache

Zerocache is a standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider. It intercepts embedding requests, serves previously-computed vectors from a local content-addressed store, and forwards only the cache misses upstream.

It is API-compatible with the OpenAI `/v1/embeddings` endpoint shape, so any TS or Python agent orchestration framework (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack, ...) can adopt it by pointing its existing embedding client at a different `base_url` — no SDK to install, no framework-specific integration code.

Multi-provider, multi-tenant: pick OpenAI, Mistral, or Gemini per request via the URL path, bringing your own API key for that provider. Zerocache holds no provider credentials of its own, and the cache is scoped per-caller — two different callers' identical requests never share a cache entry.

## Status

Early Phase 1: the Cargo workspace is scaffolded and builds/tests clean, but it has not yet been wired up to a live ingestion pipeline. See [`PRD.md`](./PRD.md) for the full product spec, phasing, and success criteria, [`CLAUDE.md`](./CLAUDE.md) for architecture notes aimed at future contributors (human or AI), and [`decisions.md`](./decisions.md) for the reasoning behind the multi-tenant, multi-provider design.

## Why

RAG ingestion pipelines re-embed text that's already been embedded before — during re-indexing, pipeline re-runs, or overlapping corpora across projects. Every re-embed is a wasted provider call: it costs input tokens and adds latency for a result that's byte-identical to something already computed. Zerocache eliminates that waste at the wire level, independent of which framework or language produced the request.

## Architecture

Dependencies point inward only, enforced via Cargo workspace crate boundaries:

| Crate | Responsibility |
| --- | --- |
| `zerocache-core` | Domain logic: `CacheKey` derivation, hit/miss reconciliation. No I/O, no async runtime. |
| `zerocache-ports` | `EmbeddingStore` / `EmbeddingProvider` trait contracts. |
| `zerocache-adapters-sled` | `EmbeddingStore` implementation backed by [sled](https://github.com/spacejam/sled) — embedded, single-process. Local dev / single-instance only. |
| `zerocache-adapters-redis` | `EmbeddingStore` implementation backed by Redis — shared, network-accessible. Use this for multi-replica (e.g. Kubernetes) deployments. |
| `zerocache-adapters-openai` | `EmbeddingProvider` implementation for OpenAI. |
| `zerocache-adapters-mistral` | `EmbeddingProvider` implementation for Mistral. |
| `zerocache-adapters-gemini` | `EmbeddingProvider` implementation for Gemini (different auth scheme — `x-goog-api-key`, not a bearer token — and a different wire shape entirely). |
| `zerocache-http` | axum HTTP server, wire-shape translation, provider registry, and application wiring. |

The cache key is `blake3(owner_id, provider, model, model_version, text)`. `owner_id` is a hash of the caller's own forwarded API key (never the raw key), scoping the cache per-caller; `provider` and model identity are included so a different provider, model, or version can never silently return a stale-but-plausible vector.

## Getting started

### Prerequisites

Rust via [rustup](https://rustup.rs) (edition 2021).

### Build & test

```sh
cargo build --workspace
cargo test --workspace
```

### Run

```sh
cargo run -p zerocache-http
```

No provider API key is configured on the server — every caller brings their own (see [API](#api) below). Configuration is environment-variable only:

| Variable | Required | Default |
| --- | --- | --- |
| `ZEROCACHE_PORT` | no | `8080` |
| `ZEROCACHE_STORAGE_BACKEND` | no | `sled` (or `redis`) |
| `ZEROCACHE_STORAGE_PATH` | no, sled only | `./data` |
| `ZEROCACHE_REDIS_URL` | no, redis only | `redis://127.0.0.1:6379` |

`ZEROCACHE_STORAGE_BACKEND=sled` (the default) is embedded and single-process — fine for local dev, but each replica would keep its own private cache. Use `redis` for any deployment with more than one instance (e.g. Kubernetes) so all replicas share one cache; it's connection-pooled with no distributed locking, since the content-addressed key means concurrent writes from different replicas are never conflicting.

## API

```text
POST /{provider}/v1/embeddings
Authorization: Bearer <your own API key for that provider>
{ "model": "<real upstream model name>", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

`{provider}` is `openai`, `mistral`, or `gemini`. Every major orchestrator (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack) configures its embedding client with exactly three knobs — `base_url`, `api_key`, `model` — so pointing at Zerocache is just `base_url: "https://<your-zerocache>/mistral"` with your own Mistral key and `model: "mistral-embed"`. No plugin, no custom headers, no body-shape change.

Missing/malformed `Authorization` → `401`. Unknown `{provider}` → `404`. The cache is scoped per-caller: two different API keys requesting the same text under the same model never share a cache entry, even though a single caller's repeated requests always do. Each response also carries `X-Zerocache-Hits` / `X-Zerocache-Misses` headers, and `usage` reflects only what was actually billed by the provider for this request (0 for an all-cache-hit batch, and always 0 for Gemini, which does not report usage at all).

### Metrics

```text
GET /metrics
```

Cumulative counters in Prometheus text exposition format, labeled by `provider`: `zerocache_cache_hits_total{provider="..."}`, `zerocache_cache_misses_total{provider="..."}`, `zerocache_provider_prompt_tokens_total{provider="..."}`. No owner/tenant label — that would leak tenant identity into a monitoring system and create unbounded cardinality. Per-instance — with multiple replicas (`ZEROCACHE_STORAGE_BACKEND=redis`), point your Prometheus scrape config at each pod and aggregate with `sum()` for a fleet-wide view.

## Non-goals (v1)

- Live/conversational query embedding caching
- Semantic/fuzzy similarity matching (exact-match only)
- Vector quantization/compression, eviction
- Multi-provider *failover* (automatic fallback to a second provider if the first fails) — multi-provider *support* itself is implemented (OpenAI, Mistral, Gemini)
- Per-tenant rate limiting or quota enforcement
- Any Zerocache-specific SDK or client package

See [`PRD.md`](./PRD.md) §4 for the full rationale.

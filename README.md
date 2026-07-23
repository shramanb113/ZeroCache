# Zerocache

Zerocache is a standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider. It intercepts embedding requests, serves previously-computed vectors from a local content-addressed store, and forwards only the cache misses upstream.

It is API-compatible with the OpenAI `/v1/embeddings` endpoint shape, so any TS or Python agent orchestration framework (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack, ...) can adopt it by pointing its existing embedding client at a different `base_url` — no SDK to install, no framework-specific integration code.

## Status

Early Phase 1: the Cargo workspace is scaffolded and builds/tests clean, but it has not yet been wired up to a live ingestion pipeline. See [`PRD.md`](./PRD.md) for the full product spec, phasing, and success criteria, and [`CLAUDE.md`](./CLAUDE.md) for architecture notes aimed at future contributors (human or AI).

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
| `zerocache-adapters-openai` | `EmbeddingProvider` implementation calling an OpenAI-compatible endpoint. |
| `zerocache-http` | axum HTTP server, OpenAI wire-shape translation, and application wiring. |

The cache key is `blake3(model_id, model_version, text)` — including model identity is deliberate, so a model or version upgrade can't silently return a stale vector that merely looks like a valid hit.

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
ZEROCACHE_PROVIDER_API_KEY=sk-... cargo run -p zerocache-http
```

Configuration is environment-variable only:

| Variable | Required | Default |
| --- | --- | --- |
| `ZEROCACHE_PROVIDER_API_KEY` | yes | — |
| `ZEROCACHE_PORT` | no | `8080` |
| `ZEROCACHE_STORAGE_BACKEND` | no | `sled` (or `redis`) |
| `ZEROCACHE_STORAGE_PATH` | no, sled only | `./data` |
| `ZEROCACHE_REDIS_URL` | no, redis only | `redis://127.0.0.1:6379` |
| `ZEROCACHE_PROVIDER_BASE_URL` | no | `https://api.openai.com` |

`ZEROCACHE_STORAGE_BACKEND=sled` (the default) is embedded and single-process — fine for local dev, but each replica would keep its own private cache. Use `redis` for any deployment with more than one instance (e.g. Kubernetes) so all replicas share one cache; it's connection-pooled with no distributed locking, since the content-addressed key means concurrent writes from different replicas are never conflicting.

## API

A single endpoint matching the standard embeddings API shape:

```text
POST /v1/embeddings
{ "model": "...", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

The response shape is preserved exactly so existing OpenAI-compatible client libraries in either language require no code change beyond the base URL. Each response also carries `X-Zerocache-Hits` / `X-Zerocache-Misses` headers, and `usage` reflects only what was actually billed by the provider for this request (0 for an all-cache-hit batch).

### Metrics

```text
GET /metrics
```

Cumulative counters in Prometheus text exposition format: `zerocache_cache_hits_total`, `zerocache_cache_misses_total`, `zerocache_provider_prompt_tokens_total`. Per-instance — with multiple replicas (`ZEROCACHE_STORAGE_BACKEND=redis`), point your Prometheus scrape config at each pod and aggregate with `sum()` for a fleet-wide view.

## Non-goals (v1)

- Live/conversational query embedding caching
- Semantic/fuzzy similarity matching (exact-match only)
- Vector quantization/compression, eviction
- Multi-provider failover, distributed storage
- Any Zerocache-specific SDK or client package

See [`PRD.md`](./PRD.md) §4 for the full rationale.

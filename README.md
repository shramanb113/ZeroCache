# Zerocache

Zerocache is a standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider. It intercepts embedding requests, serves previously-computed vectors from a local content-addressed store, and forwards only the cache misses upstream.

It is API-compatible with the OpenAI `/v1/embeddings` endpoint shape, so any TS or Python agent orchestration framework (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack, ...) can adopt it by pointing its existing embedding client at a different `base_url` — no SDK to install, no framework-specific integration code.

Multi-provider, multi-tenant: pick OpenAI, Mistral, Gemini, or HuggingFace per request via the URL path, bringing your own API key for that provider. Zerocache holds no provider credentials of its own, and the cache is scoped per-caller — two different callers' identical requests never share a cache entry. Gemini also supports image embeddings via a parallel `/gemini/v1/images/embeddings` route.

## Status

Phase 1 complete and validated against three independently-built, real consumers: a TypeScript/Mastra RAG pipeline (including an agentic battle-test driving Zerocache through `Agent` tool calls, not just a direct embedding client), a second TypeScript project on LangChain, and a Python/LlamaIndex pipeline against a different provider (Gemini) — proving the "any framework, any language" neutrality claim rather than just asserting it. Production-trust basics (provider timeouts, graceful shutdown, `/health` + `/ready`, request coalescing, retry/backoff, OpenTelemetry tracing) are in place. Cloud-provider adapters (Azure OpenAI, Amazon Bedrock, GCP Vertex AI) are in active development as of 2026-07-29 — see the Architecture table below for current per-crate status. See [`PRD.md`](./PRD.md) for the full product spec, phasing, and success criteria, [`CLAUDE.md`](./CLAUDE.md) for architecture notes aimed at future contributors (human or AI), and [`decisions.md`](./decisions.md) for the reasoning behind the multi-tenant, multi-provider design.

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
| `zerocache-adapters-gemini` | `EmbeddingProvider` and `ImageEmbeddingProvider` implementation for Gemini (different auth scheme — `x-goog-api-key`, not a bearer token — and a different wire shape entirely; the only provider with image-embedding support so far). |
| `zerocache-adapters-huggingface` | `EmbeddingProvider` implementation for HuggingFace Inference Providers. |
| `zerocache-adapters-cloud` | Shared kit for the cloud-provider adapters below: HTTP transport driver plus a `CloudRouter`/`TextWireStrategy` strategy-pattern abstraction, since each cloud is one API in front of several independent model vendors. |
| `zerocache-adapters-bedrock` | `EmbeddingProvider` implementation for Amazon Bedrock (Titan, Cohere). **Implemented.** |
| `zerocache-adapters-vertexai` | `EmbeddingProvider` implementation for GCP Vertex AI's native `:predict` endpoint. **In progress.** |
| `zerocache-adapters-azure` | `EmbeddingProvider` implementation for Azure OpenAI + Foundry Models. **Not yet implemented** — placeholder crate. |
| `zerocache-http` | axum HTTP server, wire-shape translation, provider registry, and application wiring. The three cloud adapters above are not yet registered here. |

The cache key is `blake3(owner_id, provider, cache_scope, model, model_version, text)`. `owner_id` is a hash of the caller's own forwarded API key (never the raw key), scoping the cache per-caller; `provider`, `cache_scope`, and model identity are all included so a different provider, endpoint/deployment, model, or version can never silently return a stale-but-plausible vector. `cache_scope` (added 2026-07-28) is provider-specific — for the four base adapters it's the configured base URL, so repointing one at a different endpoint (e.g. a self-hosted vLLM instance) starts from a cold cache instead of risking a wrong hit.

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
| `ZEROCACHE_TTL_SECONDS` | no | unset (entries never expire) — optional per-store-instance expiry; `0` or an unparseable value is treated as unset, with a startup warning |
| `ZEROCACHE_OPENAI_BASE_URL` | no | `https://api.openai.com` — optional override, e.g. for a self-hosted vLLM/LM Studio instance speaking OpenAI's `/v1/embeddings` wire shape. Must be a bare origin (scheme + host + optional port) with no `/v1` suffix and no trailing slash — the adapter appends its own path |
| `ZEROCACHE_MISTRAL_BASE_URL` | no | `https://api.mistral.ai` — optional override, e.g. for an on-prem Mistral-compatible proxy. Must be a bare origin with no `/v1` suffix and no trailing slash — the adapter appends its own path |
| `ZEROCACHE_GEMINI_BASE_URL` | no | `https://generativelanguage.googleapis.com` — optional override. Must be a bare origin with no `/v1` suffix and no trailing slash — the adapter appends its own path |
| `ZEROCACHE_HUGGINGFACE_BASE_URL` | no | `https://router.huggingface.co/hf-inference` — optional override, e.g. for a self-hosted HuggingFace TEI instance. Must be a bare origin with no `/v1` suffix and no trailing slash — the adapter appends its own path |

`ZEROCACHE_STORAGE_BACKEND=sled` (the default) is embedded and single-process — fine for local dev, but each replica would keep its own private cache. Use `redis` for any deployment with more than one instance (e.g. Kubernetes) so all replicas share one cache; it's connection-pooled with no distributed locking, since the content-addressed key means concurrent writes from different replicas are never conflicting.

### Demo

[`demo/`](./demo) is a real, unmodified [Mastra](https://mastra.ai) TypeScript project — not a mocked example — with a small workflow (`duplicate-finder`) that embeds a batch of text through Zerocache and flags near-duplicates by cosine similarity. It's the clearest way to see Zerocache actually work against a real framework and a real provider.

```sh
# terminal 1: Zerocache itself
cargo run -p zerocache-http

# terminal 2: the demo
cd demo
npm install
npm run dev
```

Open [http://localhost:4111](http://localhost:4111) (Mastra Studio), go to **Workflows → duplicate-finder**, and run it with a real provider API key (Gemini, using model `gemini-embedding-001`) and a batch of texts, e.g.:

```json
{
  "apiKey": "<your Gemini API key>",
  "texts": [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog.",
    "A sentence about zerocache and embedding caches"
  ]
}
```

Run it once, then run it again with the same (or an overlapping) batch — Studio's graph view shows each step's real duration, and `elapsedMs` in the output drops sharply once the texts are cached. The workflow itself never talks to Zerocache-specific code beyond a `base_url`/`apiKey`/`model` config — see [`demo/src/mastra/workflows/duplicate-finder.ts`](./demo/src/mastra/workflows/duplicate-finder.ts).

A minimal standalone version without the Mastra Studio UI is also included at [`demo/test-embed.mjs`](./demo/test-embed.mjs): `ZEROCACHE_DEMO_KEY=<your key> node demo/test-embed.mjs`.

## API

```text
POST /{provider}/v1/embeddings
Authorization: Bearer <your own API key for that provider>
{ "model": "<real upstream model name>", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

`{provider}` is `openai`, `mistral`, `gemini`, or `huggingface`. `input` accepts either a JSON array of strings or a single bare string, matching OpenAI's real `input: string | string[]` contract. Every major orchestrator (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack) configures its embedding client with exactly three knobs — `base_url`, `api_key`, `model` — so pointing at Zerocache is just `base_url: "https://<your-zerocache>/mistral"` with your own Mistral key and `model: "mistral-embed"`. No plugin, no custom headers, no body-shape change.

A matching `DELETE /{provider}/v1/embeddings` (same body shape) removes the cache entries a matching `POST` would have hit, scoped to the caller's own `owner_id`.

Missing/malformed `Authorization` → `401`. Unknown `{provider}` → `404`. Valid JSON with a missing/wrong-typed required field → `422`; malformed JSON → `400` — both come back as `{"error": "..."}`, same shape as every other error path. The cache is scoped per-caller: two different API keys requesting the same text under the same model never share a cache entry, even though a single caller's repeated requests always do. Concurrent requests that miss on the exact same key within one instance are coalesced into a single provider call. Each response also carries `X-Zerocache-Hits` / `X-Zerocache-Misses` headers, and `usage` reflects only what was actually billed by the provider for this request (0 for an all-cache-hit batch, 0 for a coalesced request that piggybacked, and always 0 for Gemini, which does not report usage at all).

### Image embeddings

```text
POST /gemini/v1/images/embeddings
Authorization: Bearer <your Gemini API key>
{ "model": "gemini-embedding-001", "input": ["data:image/png;base64,<...>", ...] }
```

Gemini only, via `AppState.image_providers` — every other provider correctly `404`s with `{"error": "provider '<name>' does not support image embeddings"}`. Same `DELETE`, error-shape, and per-caller isolation semantics as the text endpoint.

### Operational endpoints (unauthenticated)

```text
GET /health    liveness — 200 OK means only the process/router is up
GET /ready     readiness — 200 OK if the configured store backend answers a get(); 503 otherwise
GET /metrics   Prometheus text format
```

`/metrics` gives cumulative counters labeled by `provider` and `content_type` (`text`/`image`): `zerocache_cache_hits_total`, `zerocache_cache_misses_total`, `zerocache_provider_prompt_tokens_total`. No owner/tenant label — that would leak tenant identity into a monitoring system and create unbounded cardinality. Per-instance — with multiple replicas (`ZEROCACHE_STORAGE_BACKEND=redis`), point your Prometheus scrape config at each pod and aggregate with `sum()` for a fleet-wide view.

## Non-goals (v1)

- Live/conversational query embedding caching
- Semantic/fuzzy similarity matching (exact-match only)
- Vector quantization/compression, eviction
- Multi-provider *failover* (automatic fallback to a second provider if the first fails) — multi-provider *support* itself is implemented (OpenAI, Mistral, Gemini, HuggingFace; Azure/Bedrock/Vertex AI in progress)
- Per-tenant rate limiting or quota enforcement
- Any Zerocache-specific SDK or client package

See [`PRD.md`](./PRD.md) §4 for the full rationale.

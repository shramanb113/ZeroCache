# Zerocache

**A drop-in, Rust-native embedding cache.** Point your existing OpenAI-compatible embedding client at Zerocache instead of the real provider, and identical text/image inputs stop costing you a second API call.

[![CI](https://github.com/shramanb113/ZeroCache/actions/workflows/ci.yml/badge.svg)](https://github.com/shramanb113/ZeroCache/actions/workflows/ci.yml)
[![Docker Publish](https://github.com/shramanb113/ZeroCache/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/shramanb113/ZeroCache/actions/workflows/docker-publish.yml)
![Rust](https://img.shields.io/badge/rust-1.97.1-orange)
![Providers](https://img.shields.io/badge/providers-7-blue)

No SDK to install. No framework plugin. No server-side provider credentials for Zerocache to leak — every request brings its own API key.

```
Your app  ──▶  Zerocache  ──▶  Real provider (only on a cache miss)
              (drop-in base_url swap)
```

---

## Table of contents

- [Why Zerocache exists](#why-zerocache-exists)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Supported providers](#supported-providers)
- [Quickstart](#quickstart)
  - [Docker (recommended)](#docker-recommended)
  - [Docker Compose (with Redis)](#docker-compose-with-redis)
  - [From source](#from-source)
- [Configuration reference](#configuration-reference)
- [API reference](#api-reference)
  - [Text embeddings](#text-embeddings)
  - [Image embeddings](#image-embeddings-gemini-only)
  - [Provider model-string grammars](#provider-model-string-grammars-cloud-providers)
  - [Error shapes](#error-shapes)
  - [Response headers](#response-headers)
  - [Operational endpoints](#operational-endpoints)
- [Deployment](#deployment)
  - [Docker image](#docker-image)
  - [Kubernetes / multi-replica](#kubernetes--multi-replica)
  - [CI/CD pipeline](#cicd-pipeline)
- [Observability](#observability)
- [Testing](#testing)
- [Project status](#project-status)
- [Non-goals (v1)](#non-goals-v1)
- [Contributing / further reading](#contributing--further-reading)
- [License](#license)

---

## Why Zerocache exists

RAG ingestion pipelines re-embed text that's already been embedded before — during re-indexing, pipeline re-runs, CI test suites, or overlapping corpora across projects and teams. Every re-embed is pure waste: it costs real input tokens and adds real latency for a result that's byte-identical to something already computed.

Zerocache eliminates that waste at the wire level, transparently, independent of which language or framework produced the request. A real measured example from this repo's own agentic battle-test: re-indexing a 9-document corpus after editing one document and adding another cost exactly **2 provider calls out of 9** — the other 7, byte-identical between versions, were served from cache for free.

## How it works

1. Your embedding client sends a normal `POST /v1/embeddings`-shaped request — except the URL now points at Zerocache, with a provider name in the path (`/openai/v1/embeddings`, `/gemini/v1/embeddings`, etc.).
2. Zerocache derives a content-addressed cache key from `owner_id + provider + cache_scope + model + model_version + text`, using your forwarded API key (hashed, never stored raw) to keep your cache private to you.
3. Any input already in the store is returned instantly. Anything new is batched into a single call to the *real* upstream provider, using *your* forwarded key — Zerocache never sees or stores a provider credential beyond the duration of that one request.
4. New vectors are written back to the store and returned alongside the cache hits, in the original request order.
5. Concurrent requests that miss on the *exact same* text are automatically coalesced into one upstream call, not one per request.

## Architecture

Dependencies point inward only — a hard, structurally-enforced rule via Cargo workspace crate boundaries, not just convention:

```mermaid
flowchart TB
    subgraph layerInterface["Interface / Transport"]
        nodeHttp["zerocache-http<br/>axum · wire-shape translation · provider registry"]
    end
    subgraph layerApplication["Application"]
        nodeApp["orchestration: split hits/misses,<br/>call provider for misses only,<br/>write back, reassemble in order"]
    end
    subgraph layerPorts["Ports"]
        nodePorts["EmbeddingStore · EmbeddingProvider ·<br/>ImageEmbeddingProvider trait contracts"]
    end
    subgraph layerAdapters["Adapters"]
        nodeStore["Store adapters<br/>sled · redis"]
        nodeProvider["Provider adapters<br/>openai · mistral · gemini · huggingface<br/>bedrock · vertexai · azure"]
    end
    subgraph layerCore["Core (domain)"]
        nodeKey["CacheKey derivation (blake3)<br/>hit/miss reconciliation<br/>zero I/O, zero async runtime"]
    end

    nodeClient(["Any OpenAI-compatible<br/>embedding client"]) -->|"POST /{provider}/v1/embeddings"| nodeHttp
    nodeHttp --> nodeApp
    nodeApp --> nodePorts
    nodePorts --> nodeStore
    nodePorts --> nodeProvider
    nodeApp --> nodeKey
    nodeProvider -->|"BYOK: your forwarded key"| nodeUpstream(["Real provider API"])
    nodeHttp -->|"ordered response +<br/>X-Zerocache-Hits/-Misses"| nodeClient
```

| Crate | Responsibility |
| --- | --- |
| `zerocache-core` | Domain logic: `CacheKey`/`CacheKey::derive_image` derivation, hit/miss reconciliation. No I/O, no async runtime, no framework awareness. |
| `zerocache-ports` | `EmbeddingStore` / `EmbeddingProvider` / `ImageEmbeddingProvider` trait contracts, `StoreError`/`ProviderError`/`ProviderUsage`. |
| `zerocache-adapters-sled` | `EmbeddingStore` backed by [sled](https://github.com/spacejam/sled) — embedded, single-process. Local dev / single-instance. |
| `zerocache-adapters-redis` | `EmbeddingStore` backed by Redis — shared, network-accessible, connection-pooled. Use for any multi-replica deployment. |
| `zerocache-adapters-openai` | `EmbeddingProvider` for OpenAI. |
| `zerocache-adapters-mistral` | `EmbeddingProvider` for Mistral. |
| `zerocache-adapters-gemini` | `EmbeddingProvider` **and** `ImageEmbeddingProvider` for Gemini — the only provider with image-embedding support. |
| `zerocache-adapters-huggingface` | `EmbeddingProvider` for HuggingFace Inference Providers. |
| `zerocache-adapters-cloud` | Shared kit for the three cloud adapters below: HTTP transport driver (client, timeouts, retry, chunking, usage accounting) plus a `CloudRouter`/`TextWireStrategy` strategy-pattern abstraction, since each cloud is one API in front of several independent model vendors. |
| `zerocache-adapters-bedrock` | `EmbeddingProvider` for Amazon Bedrock (Titan, Cohere). |
| `zerocache-adapters-vertexai` | `EmbeddingProvider` for GCP Vertex AI's native `:predict` endpoint. |
| `zerocache-adapters-azure` | `EmbeddingProvider` for Azure — both the GA OpenAI `/openai/v1` surface and Foundry Models. |
| `zerocache-http` | axum HTTP server, wire-shape translation, provider registry, application wiring. Registers all seven provider adapters. |

The cache key is `blake3(owner_id, provider, cache_scope, model, model_version, text)`:

- **`owner_id`** — a hash of your forwarded API key (never the raw key), so two different callers never share a cache entry even for identical text.
- **`provider` + `model` + `model_version`** — so a different provider, model, or adapter version can never silently return a stale-but-plausible vector.
- **`cache_scope`** — provider-specific routing identity: the configured base URL for the four "simple" adapters (so repointing `ZEROCACHE_OPENAI_BASE_URL` at a self-hosted vLLM instance starts from a cold cache, never a wrong hit), and `{endpoint_base}\0{canonical}\0kit{version}` for the three cloud adapters (so `us-east-1` and `eu-west-1` Bedrock, or two different GCP Vertex AI projects, can never collide even when the caller's `model` string looks identical).

## Supported providers

| `{provider}` | Text embeddings | Image embeddings | Auth to Zerocache | Notes |
| --- | :---: | :---: | --- | --- |
| `openai` | ✅ | — | `Authorization: Bearer <key>` | Configurable base URL — self-hosted vLLM/LM Studio work too. |
| `mistral` | ✅ | — | `Authorization: Bearer <key>` | Configurable base URL. |
| `gemini` | ✅ | ✅ | `Authorization: Bearer <key>` | Only provider with image embeddings. Never reports token usage. |
| `huggingface` | ✅ | — | `Authorization: Bearer <key>` | Model is part of the URL path, not the JSON body — a genuine wire-shape difference from the other three. |
| `bedrock` | ✅ | — | `Authorization: Bearer <key>` | Amazon's own bearer API keys — no AWS SigV4. Titan and Cohere vendor models behind one router. |
| `vertexai` | ✅ | — | `Authorization: Bearer <key>` | GCP's native `:predict` endpoint (Vertex's OpenAI-compatible surface is chat-only). |
| `azure` | ✅ | — | `Authorization: Bearer <key>` | Two surfaces in one adapter: Azure OpenAI GA `/openai/v1` and Foundry Models — routed by a `foundry:` model prefix. Only registers if at least one of its two base-URL env vars is set. |

Every caller brings their own key for whichever provider they call — Zerocache holds no provider credentials of its own (BYOK: bring-your-own-key).

## Quickstart

### Docker (recommended)

```sh
docker run -d --name zerocache -p 8080:8080 ghcr.io/shramanb113/zerocache:latest
```

That's it — no provider key needed to start the container; you supply one per request (see [API reference](#api-reference)). Confirm it's up:

```sh
curl http://localhost:8080/health
```

The image runs as a non-root user, ships a built-in `HEALTHCHECK` against `/health`, and defaults to the embedded `sled` store at `/data` inside the container — mount a volume there if you want the cache to survive a restart:

```sh
docker run -d --name zerocache -p 8080:8080 -v zerocache-data:/data ghcr.io/shramanb113/zerocache:latest
```

> **Bind mounts vs named volumes:** the container runs as uid `10001`. A **named volume** (`-v zerocache-data:/data`, as above) inherits the image's ownership and just works. A **bind mount** to a host directory (`-v ./local-data:/data`) takes the *host* directory's ownership instead — if that directory isn't writable by uid `10001`, sled fails to open and the process exits. Prefer named volumes unless you specifically need the data on a host path, in which case `chown -R 10001:10001` that directory first.

### Docker Compose (with Redis)

For local development against the Redis backend (what you'd run in a multi-replica deployment):

```sh
docker compose up -d --build
```

This brings up `zerocache-http` wired to a `redis` service (`ZEROCACHE_STORAGE_BACKEND=redis`), both on the same compose network. See [`docker-compose.yml`](./docker-compose.yml).

### From source

Prerequisites: Rust via [rustup](https://rustup.rs), toolchain `1.97.1` (edition 2021).

```sh
cargo build --workspace
cargo test --workspace
cargo run -p zerocache-http
```

Real-Redis integration tests are `#[ignore]`d by default (they spin up an ephemeral container via `testcontainers`, so they need Docker running):

```sh
cargo test -p zerocache-adapters-redis -- --ignored
```

## Configuration reference

Every setting is an environment variable — there is no config file. Everything is optional; Zerocache starts with sensible defaults and zero provider keys.

**Core**

| Variable | Default | Notes |
| --- | --- | --- |
| `ZEROCACHE_PORT` | `8080` | HTTP listen port (binds `0.0.0.0`). |
| `ZEROCACHE_STORAGE_BACKEND` | `sled` | `sled` or `redis`. |
| `ZEROCACHE_STORAGE_PATH` | `./data` (`/data` in the Docker image) | sled only. |
| `ZEROCACHE_REDIS_URL` | `redis://127.0.0.1:6379` | redis only. |
| `ZEROCACHE_TTL_SECONDS` | unset (never expires) | Per-store-instance expiry. `0` or an unparseable value is treated as unset, with a startup warning. |

**Simple provider base-URL overrides** — a bare origin (scheme + host + optional port), **no** `/v1` suffix, **no** trailing slash:

| Variable | Default |
| --- | --- |
| `ZEROCACHE_OPENAI_BASE_URL` | `https://api.openai.com` |
| `ZEROCACHE_MISTRAL_BASE_URL` | `https://api.mistral.ai` |
| `ZEROCACHE_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` |
| `ZEROCACHE_HUGGINGFACE_BASE_URL` | `https://router.huggingface.co/hf-inference` |

**Azure**

| Variable | Default | Notes |
| --- | --- | --- |
| `ZEROCACHE_AZURE_OPENAI_BASE_URL` | unset | e.g. `https://my-resource.openai.azure.com`. Setting **either** this or the Foundry URL below registers the `azure` provider. |
| `ZEROCACHE_AZURE_FOUNDRY_BASE_URL` | unset | e.g. `https://my-resource.services.ai.azure.com`. |
| `ZEROCACHE_AZURE_FOUNDRY_API_VERSION` | `2024-05-01-preview` | Foundry surface only — the GA `/openai/v1` path takes no api-version. |
| `ZEROCACHE_AZURE_AUTH_MODE` | `bearer` | `bearer` (Entra ID token, recommended) or `api-key`. An unrecognized value warns and falls back to `bearer`. |

**Amazon Bedrock**

| Variable | Default |
| --- | --- |
| `ZEROCACHE_BEDROCK_REGION` | `us-east-1` |
| `ZEROCACHE_BEDROCK_ENDPOINT_TEMPLATE` | `https://bedrock-runtime.{region}.amazonaws.com` |

**GCP Vertex AI**

| Variable | Default | Notes |
| --- | --- | --- |
| `ZEROCACHE_VERTEX_PROJECT` | unset | If unset, every `vertexai` request's `model` must carry `<location>/<project>/` itself. |
| `ZEROCACHE_VERTEX_LOCATION` | `us-central1` | |
| `ZEROCACHE_VERTEX_ENDPOINT_TEMPLATE` | `https://{location}-aiplatform.googleapis.com` | `global`/`us`/`eu` multi-region locations resolve to the correct real Google host automatically. |

**Observability** (optional, off by default)

| Variable | Default |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset — no OTLP export, console logging only |
| `RUST_LOG` | `info` |

## API reference

### Text embeddings

```
POST /{provider}/v1/embeddings
Authorization: Bearer <your own API key for that provider>
Content-Type: application/json

{ "model": "<real upstream model name>", "input": ["text 1", "text 2"] }
```

```
→ 200 OK
{
  "object": "list",
  "data": [ { "embedding": [0.001, -0.02, ...], "index": 0 }, ... ],
  "model": "...",
  "usage": { "prompt_tokens": 12, "total_tokens": 12 }
}
```

`{provider}` is one of `openai`, `mistral`, `gemini`, `huggingface`, `bedrock`, `vertexai`, `azure`. `input` accepts either a JSON array of strings or a single bare string, matching OpenAI's real `input: string | string[]` contract — so `embedQuery()`-style single-string calls (LangChain, LlamaIndex, etc.) work without modification.

Example, OpenAI:
```sh
curl https://your-zerocache-host/openai/v1/embeddings \
  -H "Authorization: Bearer sk-your-real-openai-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "hello world"}'
```

Example, Bedrock (region + Cohere input_type encoded in `model`):
```sh
curl https://your-zerocache-host/bedrock/v1/embeddings \
  -H "Authorization: Bearer your-bedrock-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "us-east-1/cohere.embed-english-v3#search_query", "input": ["find me something"]}'
```

A matching `DELETE /{provider}/v1/embeddings` (identical body shape) removes the cache entries a matching `POST` would have hit, scoped to your own `owner_id`. Response: `{"deleted": <count>}` — the count of keys *requested*, not how many actually existed (deletion is idempotent).

### Image embeddings (Gemini only)

```
POST /gemini/v1/images/embeddings
Authorization: Bearer <your Gemini API key>
Content-Type: application/json

{ "model": "gemini-embedding-001", "input": ["data:image/png;base64,<...>", ...] }
```

Every other provider `404`s on this route with `{"error": "provider '<name>' does not support image embeddings"}`. Same `DELETE`, error-shape, and per-caller isolation semantics as the text endpoint.

### Provider model-string grammars (cloud providers)

Azure, Bedrock, and Vertex AI encode routing coordinates into the `model` string itself rather than a new wire field — `model` is already free-form per-request and already part of the cache key, so this needs no framework-specific changes on the client side:

| `{provider}` | Grammar | Example |
| --- | --- | --- |
| `bedrock` | `[<region>/]<modelId>[#<input_type>]` | `us-east-1/cohere.embed-english-v3#search_query` |
| `vertexai` | `[<location>/<project>/]<modelId>[#<task_type>]` | `us-central1/my-proj/text-embedding-005#RETRIEVAL_DOCUMENT` |
| `azure` | `[foundry:]<deployment>[#<input_type>]` | `foundry:cohere-embed-v3-english#document` |

The `#<input_type>`/`#<task_type>` qualifier is included deliberately: for the vendors that accept it, it's a required-or-near-required parameter that *changes the output vector* (e.g. document vs. query embeddings). Hardcoding one value would silently give every caller the same embedding style regardless of use case — a failure mode that's invisible until retrieval quality degrades. It's caller-controlled and folded into the cache key so document- and query-style embeddings of the same text are never confused with each other.

### Error shapes

| Condition | Status | Body |
| --- | --- | --- |
| Missing/malformed `Authorization` | `401` | `{"error": "..."}` |
| Unknown `{provider}` | `404` | `{"error": "..."}` |
| Malformed JSON | `400` | `{"error": "..."}` |
| Valid JSON, missing/wrong-typed field | `422` | `{"error": "..."}` |
| Malformed image data URI | `400` | `{"error": "..."}` |

Every error path — including axum's own body-rejection errors — returns the same `{"error": "..."}` shape, never a bare plain-text response.

### Response headers

- `X-Zerocache-Hits` / `X-Zerocache-Misses` — counts for this request's batch.
- `usage` in the body reflects only what was **actually billed** for this request: `0` for an all-hit batch, `0` for a request that piggybacked on another in-flight identical request (in-process coalescing — see below), and always `0` for Gemini, which never reports token usage on any endpoint.

Concurrent requests that miss on the *exact same* cache key within one Zerocache instance are coalesced into a single upstream call — proven with a dedicated test asserting exactly one provider call across 5 genuinely-overlapping concurrent requests. This is in-process only; two different replicas behind a load balancer each still fetch independently (cross-replica coalescing would need a distributed lock and isn't built).

### Operational endpoints (unauthenticated, outside the versioned API)

```
GET /health    liveness — 200 OK means only the process/router is up, zero I/O
GET /ready     readiness — 200 OK if the configured store answers a get(); 503 otherwise
GET /metrics   Prometheus text format
```

## Deployment

### Docker image

Built from the repo-root [`Dockerfile`](./Dockerfile): a multi-stage build using [`cargo-chef`](https://github.com/LukeMathWalker/cargo-chef) for dependency-layer caching (so a source-only change doesn't force a full dependency recompile), a `debian:bookworm-slim` runtime stage, a non-root `zerocache` user (uid `10001`), and a `curl`-based `HEALTHCHECK` against `/health` that respects a custom `ZEROCACHE_PORT`. Published to **GitHub Container Registry**:

```sh
docker pull ghcr.io/shramanb113/zerocache:latest
# or pin to an exact commit:
docker pull ghcr.io/shramanb113/zerocache:<commit-sha>
```

### Kubernetes / multi-replica

`ZEROCACHE_STORAGE_BACKEND=sled` (the default) is embedded and single-process — each replica keeps its own private cache, which is fine for a single instance but means replicas never share hits. For any deployment with more than one instance, set `ZEROCACHE_STORAGE_BACKEND=redis` and point `ZEROCACHE_REDIS_URL` at a shared Redis: it's connection-pooled with no distributed locking, since content-addressed keys mean two replicas racing to fill the same key both compute the same value — a last-write-wins `SET` is always safe.

`GET /health` / `GET /ready` are wired for standard liveness/readiness probes; `GET /metrics` is Prometheus text format with `provider`/`content_type` labels — scrape every pod and aggregate with `sum()` for a fleet-wide hit rate.

### CI/CD pipeline

Two GitHub Actions workflows, both under [`.github/workflows/`](./.github/workflows/):

- **[`ci.yml`](./.github/workflows/ci.yml)** — on every push and pull request to `master`: `cargo build --workspace`, `cargo test --workspace`, the real-Redis integration suite (`cargo test -p zerocache-adapters-redis -- --ignored`, using the runner's built-in Docker), `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo fmt --check` — five independent, individually-required jobs.
- **[`docker-publish.yml`](./.github/workflows/docker-publish.yml)** — after `ci.yml` succeeds on a genuine push to `master` (explicitly not on pull requests, closing off a fork-PR path to an unreviewed publish), builds and pushes the image to `ghcr.io/shramanb113/zerocache`, tagged with both `latest` and the commit SHA.

## Observability

- **`GET /health`** — zero-I/O liveness. `200` means only "the process and router are up."
- **`GET /ready`** — real readiness: calls the configured store's `get()` against a reserved sentinel key. `200` on a miss (healthy — the key was never written), `503` on a genuine store-level error.
- **`GET /metrics`** — Prometheus counters, labeled by `provider` and `content_type` (`text`/`image`): `zerocache_cache_hits_total`, `zerocache_cache_misses_total`, `zerocache_provider_prompt_tokens_total`. Deliberately no owner/tenant label — that would leak tenant identity into a monitoring system and create unbounded cardinality.
- **OpenTelemetry tracing** — set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable OTLP/gRPC export; unset means console-only logging, no collector required to run locally. Every HTTP request gets its own span, with `store_lookup`/`provider_call`/`store_write_back` nested underneath and `hits`/`misses`/`claimed`/`piggybacked` recorded as fields.

## Testing

Ordered so cheap, deterministic layers run first:

1. **Core** — pure unit tests, no I/O (key derivation, owner/provider/cache-scope isolation, image domain-separation).
2. **Application** — orchestration logic against mock ports (hit/miss splitting, ordering, coalescing, within-batch dedup, failure propagation).
3. **Adapters** — `sled` against a real embedded store; every provider adapter against a stubbed HTTP server (`httpmock`); Redis against a genuine ephemeral container via `testcontainers` (`#[ignore]`d by default so the documented `cargo test --workspace` needs no external services — run explicitly with `-- --ignored`).
4. **End-to-end, real consumers** — not synthetic examples: a TypeScript/Mastra RAG pipeline (including an *agentic* battle-test driving Zerocache through `Agent` tool calls, not a direct embedding client), a second independent TypeScript project on LangChain, and a Python/LlamaIndex pipeline against a different provider (Gemini) — proving the "any framework, any language" neutrality claim rather than just asserting it.

At the time of writing: **196 tests passing + 7 real-Redis integration tests**, zero `cargo clippy -- -D warnings` findings, across 13 crates.

The three cloud provider adapters (Azure, Bedrock, Vertex AI) ship **mock-only** — none has had a live-key smoke test, since real credentials for those three clouds aren't available in this project's development environment. Every wire shape was verified directly against each vendor's own current documentation at implementation time, with a follow-up re-verification pass that caught and fixed a real stale-docs defect (a wrong Vertex AI endpoint-host derivation for `global`/`us`/`eu` locations). Treat this caveat as real: if you deploy one of these three, a first live smoke test against your own credentials before production traffic is a reasonable precaution.

## Project status

**Phase 1 complete.** Validated against three independently-built, real consumers across two languages (see [Testing](#testing) above). Production-trust basics are in place: provider timeouts, graceful shutdown (`SIGTERM`-aware), `/health` + `/ready`, request coalescing, retry/backoff with exponential backoff, OpenTelemetry tracing. All seven provider adapters (OpenAI, Mistral, Gemini, HuggingFace, Azure, Bedrock, Vertex AI) are implemented and registered. Docker image and CI/CD pipeline are live.

See [`PRD.md`](./PRD.md) for the full product spec and success criteria, [`CLAUDE.md`](./CLAUDE.md) for the complete architecture and decision log (every deviation from the original spec, with rationale), and [`decisions.md`](./decisions.md) for the reasoning behind the multi-tenant, multi-provider, BYOK design.

## Non-goals (v1)

- Live/conversational query embedding caching.
- Semantic/fuzzy similarity matching — exact-match only.
- Vector quantization/compression, eviction.
- Multi-provider *failover* (automatic fallback to a second provider if the first fails) — multi-provider *support* itself is fully implemented; failover is a different, separate feature.
- Per-tenant rate limiting or quota enforcement.
- Any Zerocache-specific SDK or client package — if a consumer needs to install one, the neutrality goal has failed.

See [`PRD.md`](./PRD.md) §4 for the full rationale.

## Contributing / further reading

- [`CLAUDE.md`](./CLAUDE.md) — architecture notes and a full, dated log of every deviation from the original spec, aimed at any future contributor (human or AI) who needs to understand *why* the code looks the way it does before changing it.
- [`decisions.md`](./decisions.md) — the reasoning behind major design calls (multi-tenancy, BYOK, storage backend choice, the cloud-adapter strategy pattern).
- [`PRD.md`](./PRD.md) — the original product spec and phasing.

Development loop:

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

All four are exactly what CI runs on every push and pull request — a green local run is a strong (though not complete, since the real-Redis job also needs Docker) predictor of a green CI run.

## License

No license file exists in this repository yet — until one is added, treat the code as all-rights-reserved rather than assuming permissive reuse.

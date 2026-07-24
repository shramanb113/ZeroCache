# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The Cargo workspace is scaffolded and builds/tests clean (8 crates). Treat `PRD.md` as the spec of intent, but **do not assume the port trait signatures or scope in PRD §6.2/§4 are verbatim in the code** — see "Deviations from the PRD" below; check `zerocache-ports/src/lib.rs` directly when in doubt. No CI workflow or Dockerfile exist yet (deliberately deferred — see Phasing).

### Commands

```sh
cargo build --workspace
cargo test --workspace
cargo run -p zerocache-http    # no server-side API key needed -- every caller brings their own
```

Rust toolchain: rustup-installed, verified with 1.97.1. On this dev machine `~/.cargo/bin` is not on the default shell PATH (git-bash or PowerShell) — prefix commands with it or export PATH if `cargo`/`rustc` aren't found.

## What Zerocache is

A standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider (OpenAI-compatible endpoint). It intercepts `/v1/embeddings` requests, serves previously-computed vectors from a local content-addressed store, and forwards only cache misses upstream. Adoption requires no SDK — a consumer just points its existing OpenAI-compatible embedding client at Zerocache's `base_url`.

First validation consumer: Argus (Mastra/TS + pgvector). A second, Python-based consumer (LangChain/LlamaIndex) is required before the "any framework" neutrality claim is considered proven, not just asserted (PRD §5, §14, §15).

## Architecture — layered, dependencies point inward only

This is a hard, structurally-enforced rule (via Cargo workspace crate boundaries, not convention): outer layers know about inner layers; inner layers never know outer layers exist.

| Layer | Responsibility | Depends on | Knows about I/O? |
| --- | --- | --- | --- |
| Core (domain) | Cache key derivation, hit/miss reconciliation | nothing | No |
| Ports | Trait contracts for storage, provider, wire-shape access | Core | No (traits only) |
| Application | Orchestrates end-to-end cache flow via ports | Core, Ports | No (works through traits) |
| Adapters | Concrete implementations of ports | Ports | Yes |
| Interface/Transport | HTTP surface, wire-shape translation, response shaping | Application | Yes |

### Core (domain)

Pure logic — no async runtime, no network, no disk access, nothing from `tokio`/`axum`/any storage crate.

- `CacheKey = blake3(model_id + model_version + text)`. Model identity is included deliberately so a model/version upgrade can't silently return a stale-but-plausible vector.
- Hit/miss reconciliation: given a batch of keys and lookup results, determine hits vs. misses while preserving original input ordering.
- Core has zero awareness of which framework/language sent the request — that's resolved entirely at the transport layer before anything reaches core. This is what makes "any framework" an architectural property, not a marketing claim.

### Ports

Trait contracts that core/application depend on but never implement directly (actual code, `zerocache-ports/src/lib.rs` — this differs from the PRD §6.2 snippet, see "Deviations" below):

```rust
trait EmbeddingStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Result<Option<Vec<f32>>, StoreError>;
    fn put(&self, key: CacheKey, vector: Vec<f32>) -> Result<(), StoreError>;
}

#[async_trait::async_trait]
trait EmbeddingProvider: Send + Sync {
    async fn embed_batch(&self, api_key: &str, model: &str, texts: &[String]) -> Result<(Vec<Vec<f32>>, ProviderUsage), ProviderError>;
    fn version(&self) -> &'static str;
}
```

`EmbeddingProvider` is async (2026-07-24) — adapters use async `reqwest`, not `reqwest::blocking`, so a provider call is `.await`ed directly in `zerocache-http` rather than needing `tokio::task::spawn_blocking`. `EmbeddingStore` stays synchronous (`sled`/`redis` are blocking APIs); their calls still run inside `spawn_blocking`. `version()` replaces the old global `model_version: String` on `AppState` — each adapter returns its own `env!("CARGO_PKG_VERSION")`, so the cache key's version component is tied to that adapter crate's own `Cargo.toml` version, not a manually-maintained string someone has to remember to bump. Each adapter also chunks its input into batches of `MAX_BATCH_SIZE = 100` — a single conservative constant, not tuned per-provider, since the real per-provider limits couldn't be reliably verified.

### Application

Runs the actual cache flow against the two ports: split batch into hits/misses, call the provider port for misses only, write results back through the store port, reassemble the final ordered response. Fully testable with mock ports — no real network/disk required. Also unaware of wire shape; that translation happens once, at the transport boundary.

### Adapters

Swappable concrete implementations, each independent:

- `EmbeddingStore`: two implementations exist. `zerocache-adapters-sled` (embedded, raw fp32, no compression, no eviction — content-addressed keys mean a hit is valid indefinitely, so eviction is a capacity concern only, never a correctness one) for local dev / single-instance. `zerocache-adapters-redis` (pooled via r2d2, no mutex, no distributed lock — content-addressing means two replicas racing to fill the same key both compute the same value, so a last-write-wins `SET` is safe) for multi-replica/Kubernetes deployments. Selected at startup via `ZEROCACHE_STORAGE_BACKEND=sled|redis`.
- `EmbeddingProvider`: v1 = single OpenAI-compatible adapter. Future adapters (separate crates, no change to core/ports/application): Mistral, Gemini, Voyage AI, Cohere.

Adapters depend inward on port traits only — adding a new adapter never touches core or application.

### Interface/Transport

Two deliberately separate responsibilities:

- **Wire-shape translation**: parse the incoming request (v1: OpenAI `/v1/embeddings` shape — TS and Python frameworks alike default their embedding clients to it) into the transport-agnostic internal request format.
- **HTTP surface**: an axum server handling concurrent requests from multiple independent consumers.

Keeping these separate is what lets a second wire shape be added later as just a new translation function, invisible to application/core.

## Workspace structure

```text
zerocache-core             domain: CacheKey, owner_id derivation, reconciliation — no dependencies
zerocache-ports            trait definitions (incl. StoreError/ProviderError/ProviderUsage) — depends on core
zerocache-adapters-sled    EmbeddingStore impl, embedded/single-instance — depends on ports
zerocache-adapters-redis   EmbeddingStore impl, shared/multi-instance — depends on ports
zerocache-adapters-openai  EmbeddingProvider impl (OpenAI) — depends on ports
zerocache-adapters-mistral EmbeddingProvider impl (Mistral) — depends on ports
zerocache-adapters-gemini  EmbeddingProvider impl (Gemini) — depends on ports
zerocache-http             axum transport, wire-shape translation, provider registry, application wiring — depends on all of the above
```

`zerocache-adapters-redis` is not in the PRD §13 list — see "Deviations" below. Neither are `zerocache-adapters-mistral`/`zerocache-adapters-gemini` — PRD §6.4 lists them as "Future" work; see the BYOK deviation entry above and `decisions.md`.

## Deviations from the PRD

The PRD (§3) is authoritative for intent, but treat these as the actual code, not what the literal PRD text says:

1. **`EmbeddingProvider::embed_batch` takes a `model: &str` parameter.** PRD §6.2's snippet omits it, but the wire contract (§10) requires honoring whatever `model` the client sends per-request — without it, one adapter instance could only ever serve a single hardcoded model.
2. **`EmbeddingStore`/`EmbeddingProvider` carry `Send + Sync` bounds**, needed because `zerocache-http` shares them across async axum handlers behind `Arc<dyn Trait>`.
3. **Both traits return `Result`, not bare values.** Originally scaffolded without this (matching PRD §6.2 literally) and it was a real gap — failures panicked instead of surfacing as HTTP errors. Fixed 2026-07-23.
4. **`zerocache-adapters-redis` exists in v1.** PRD §4 and §6.4 explicitly defer distributed/multi-instance storage to "Future" work. Added ahead of that schedule because Kubernetes multi-replica deployment is a real near-term requirement, not a hypothetical — confirmed with the PRD's author before building. `zerocache-adapters-sled` remains the default; Redis is opt-in via `ZEROCACHE_STORAGE_BACKEND=redis`. An optional `ZEROCACHE_TTL_SECONDS` (unset by default, meaning entries never expire) sets a per-store-instance expiry: Redis uses native `SET...EX`, sled stores an 8-byte expiry-timestamp prefix ahead of each vector and checks it lazily on read. `0` or an unparseable value is treated as unset (with a startup warning), since `0` behaves inconsistently between backends (Redis rejects it, sled would treat it as instant-expiry).
5. **`GET /metrics` exists in v1**, ahead of PRD §11's full observability spec. Prometheus text format, three counters (`zerocache_cache_hits_total`, `zerocache_cache_misses_total`, `zerocache_provider_prompt_tokens_total`) — deliberately scoped to just what Phase 1's success criteria (§15: measured hit rate, measured tokens billed) need, not per-consumer tagging or latency-saved-vs-baseline, which need decisions not yet made. Chosen over a bespoke JSON endpoint specifically because with `zerocache-adapters-redis` multi-replica deployments, per-pod counters only mean something once something aggregates across pods — Prometheus's scrape-and-`sum()` model does that; a single pod's JSON response can't.
6. **v1's single operator-configured provider (`ZEROCACHE_PROVIDER_API_KEY`) is gone, replaced by per-request bring-your-own-key (2026-07-24).** PRD §6.4/§4 describe additional provider adapters as "Future" work behind one operator-configured provider — this goes further: any of three providers (`openai`, `mistral`, `gemini`) can be selected per-request via the URL path, using the caller's own forwarded key, never Zerocache's. This also activates PRD §7's previously-deferred namespacing: `CacheKey` now includes an `owner_id` (a hash of the caller's key, never the raw key) alongside `provider`, `model`, `model_version`, `text`. Full reasoning in `decisions.md`.
7. **`EmbeddingProvider` is async (2026-07-24).** Adapters use async `reqwest`, removing the earlier `reqwest::blocking`-inside-`spawn_blocking` overhead (two runtimes, a thread-pool hop per request) — this was a real architectural wart, not a style choice; see the note in memory/commit history about why `main()` originally couldn't be `#[tokio::main]`. It can be again now. Each adapter also chunks input into batches of `MAX_BATCH_SIZE = 100` (a single conservative constant, not a verified per-provider limit — see `decisions.md`), and `version()` replaces the old global `model_version` string, tying cache-key versioning to each adapter crate's own `Cargo.toml` version instead of a manually-bumped convention.
8. **Production-trust Tier 1 (2026-07-24): provider timeouts, graceful shutdown, `/health`+`/ready`, store-call timeouts.** All three provider adapters now build their `reqwest::Client` with a uniform, unmeasured `PROVIDER_TIMEOUT = 30s` (same rationale as `MAX_BATCH_SIZE`: no verified per-provider SLA exists to tune to) — previously `reqwest::Client::new()` had no timeout at all, so a hung upstream connection blocked a request forever. `zerocache-http`'s server now shuts down gracefully via `axum::serve(...).with_graceful_shutdown(shutdown_signal())`, resolving on Ctrl+C (all platforms) or, `#[cfg(unix)]` only, `SIGTERM` (what Kubernetes sends before force-killing a pod) — **verified for real, not just read for correctness: built and ran natively on Linux via WSL2 Ubuntu, sent a genuine `kill -TERM`, confirmed the shutdown log line, a clean `exit code 0`, and no dropped in-flight request.** New unauthenticated `GET /health` (zero I/O, proves only that the process/router is up) and `GET /ready` (calls `check_store_readiness`, which reuses the existing `EmbeddingStore::get` — no new port-trait method — against a fixed reserved sentinel key that's never written to; a miss is healthy, only a store-level error means not-ready) join `/metrics` as infrastructure-facing endpoints outside the versioned API contract. Store calls (`sled`/`redis` `get`/`put`/`delete`, run via `spawn_blocking`) are now also bounded: `zerocache-http/src/app.rs`'s `run_store_task` wraps every store call in a 5s `tokio::time::timeout` uniformly (any backend degrades into a fast `AppError::Store` instead of an indefinite hang — this is what makes `/ready` fail fast instead of hanging right along with a wedged store, and keeps a stuck store call from stalling the graceful-shutdown drain); `zerocache-adapters-redis` additionally sets real socket-level `set_read_timeout`/`set_write_timeout` (5s) on every checked-out connection, which is the layer that actually unblocks a stale TCP connection at the OS level rather than just bounding how long the caller waits for it.

## API contract (v2 — multi-tenant, multi-provider)

Endpoint includes the provider in the path, and requires the caller's own provider API key — Zerocache holds no provider credentials of its own:

```text
POST /{provider}/v1/embeddings
Authorization: Bearer <caller's real provider API key>
{ "model": "<real upstream model name>", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

`{provider}` is one of `openai`, `mistral`, `gemini` for this pass (see `decisions.md` for the full reasoning and what's deliberately deferred). Missing/malformed `Authorization` header → `401`. Unknown `{provider}` → `404`. The cache is namespaced per-caller: two different callers' identical requests never share a cache entry. Responses also carry `X-Zerocache-Hits` / `X-Zerocache-Misses` headers; `usage` reflects only tokens actually billed for this request's misses (0 for an all-hit batch, and always 0 for Gemini, which does not report usage at all).

A matching `DELETE /{provider}/v1/embeddings` (same body shape: `{"model": "...", "input": [...]}`, same `Authorization` requirement) removes the cache entries a matching `POST` would have hit, scoped to the caller's own owner_id exactly like every other operation. Response: `{"deleted": <count>}`, where the count is how many keys were requested for deletion, not how many actually existed (deletion is idempotent).

### Operational endpoints (unauthenticated, outside the versioned contract)

- `GET /metrics` — Prometheus text format (see Deviations item 5).
- `GET /health` — liveness. Zero I/O; `200 OK` means only "the process and axum router are up," not that the store or any provider is reachable.
- `GET /ready` — readiness. `200 OK` if the configured store backend answers a `get()` on a reserved sentinel key; `503 SERVICE_UNAVAILABLE` on a store-level error. Does not check provider reachability (providers are BYOK and per-request, so there's no single provider connection to probe at startup).

None of the three require `Authorization` — they're infrastructure-facing, not part of `/{provider}/v1/embeddings`.

## Multi-consumer design (PRD §7, superseded 2026-07-24 — see `decisions.md`)

PRD §7 originally assumed a single global key space with namespacing/auth deferred until a second consumer needed isolation. That's no longer the current design:

- All consumers go through one running Zerocache process over HTTP — they never touch the store directly. Store concurrency is Zerocache's problem to solve once.
- Cache key is content-addressed AND owner-scoped (`owner_id + provider + model_id + model_version + text`). Two callers embedding identical text under the same model and same forwarded key share a hit for free (the "overlapping corpora across projects" benefit PRD §2 names); two different callers never do, by design — see `decisions.md` for why (cost fairness and a cache-timing existence-leak risk in a fully shared cache, once "caller" stopped being one trusted org).
- Auth is no longer deferred: every request requires a real forwarded provider key (`Authorization: Bearer <key>`), which doubles as both the credential Zerocache uses to call upstream and the input to the owner-scoping hash. There is no "trusted network boundary, no auth" mode anymore.

## Non-goals for v1 (do not implement without a scope discussion)

- Live/conversational query embedding caching (deferred — low reuse rate, unproven).
- Semantic/fuzzy similarity matching — v1 is exact-match only.
- Vector quantization/compression — deferred until a real hit-rate number justifies it.
- Multi-provider *failover* (automatic fallback if a provider call fails) — still out of scope. (Multi-*provider support* itself is no longer a non-goal, nor is multi-instance *storage* — see "Deviations" above; `zerocache-adapters-redis`, `-mistral`, `-gemini` all exist. Failover specifically — trying a second provider after the first fails — is a different feature, not built.)
- Any framework-specific SDK/client package in TS or Python — if a consumer needs to install a Zerocache-specific package, the neutrality goal has failed.

## Testing strategy (PRD §12) — order matters

1. **Core**: unit tests, no I/O — key derivation and reconciliation verified in isolation first. Done (`zerocache-core`, 10 tests, incl. owner_id/provider isolation cases).
2. **Application**: tested against mock `EmbeddingStore`/`EmbeddingProvider` — verifies orchestration without real network/disk. Done (`zerocache-http/src/app.rs` `#[cfg(test)]` module, 7 tests: hit/miss splitting, ordering, store/provider failure propagation, metrics-only-recorded-on-success, different-owners-never-share-a-cache-entry).
3. **Adapters**: integration tests against real sled and a stubbed provider. Done — `sled` roundtrip test, plus `httpmock`-stubbed tests for all three provider adapters: `zerocache-adapters-openai` (3), `zerocache-adapters-mistral` (3), `zerocache-adapters-gemini` (4, including auth-header-translation and response-count-mismatch checks). Real Redis is also covered now (2026-07-24): `zerocache-adapters-redis` has 7 `#[ignore]`d tests (roundtrip, delete, delete-of-missing-key, TTL expiry, TTL-still-valid, no-TTL-never-expires, socket-timeouts-don't-break-normal-ops) that spin up a genuine, ephemeral Redis via `testcontainers`/`testcontainers-modules` (dev-dependencies only) — each test starts its own container for full isolation. `#[ignore]`d specifically so the documented `cargo test --workspace` command keeps its "no server-side dependency needed" promise; run them explicitly with `cargo test -p zerocache-adapters-redis -- --ignored` (requires Docker running locally or in CI). All 7 pass against a real container as of this writing. Real-provider behavior additionally verified manually via the local smoke test.
4. **End-to-end, consumer 1**: real batch run through Argus (TS/Mastra) — produces Phase 1 hit-rate/cost numbers. **Not started** — Argus is a separate, large project; wiring it up is out of scope until Argus itself is ready. A manual local smoke test (real OpenAI key, miss then repeat-hit) substituted for this on 2026-07-23 and caught a real startup panic, fixed in commit `714944b`.
5. **End-to-end, consumer 2**: real batch run through a second, Python-based ingestion script (LangChain/LlamaIndex) — this is the test that actually validates the neutrality claim, not #4. Not started, blocked on #4.

## Phasing (PRD §16)

- **Phase 1 (current)**: workspace setup, core logic, port traits, sled + single-provider adapters, HTTP wire-up against Argus, first hit-rate/cost measurement.
- **Phase 1.5**: wire up a second, Python-based consumer against the same instance for a second measurement.
- **Phase 2**: scoped only after Phase 1/1.5 results are in — quantization, eviction, namespacing, additional adapters. Do not pre-build these.
- **Distribution** (Docker image, prebuilt binaries, npm/pip wrapper launchers, crates.io) is explicitly sequenced *after* Phase 1 and 1.5 validation — do not add packaging work before there's a proven hit-rate number behind it.

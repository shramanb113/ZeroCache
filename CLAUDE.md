# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The Cargo workspace is scaffolded and builds/tests clean (6 crates). Treat `PRD.md` as the spec of intent, but **do not assume the port trait signatures or scope in PRD §6.2/§4 are verbatim in the code** — see "Deviations from the PRD" below; check `zerocache-ports/src/lib.rs` directly when in doubt. No CI workflow or Dockerfile exist yet (deliberately deferred — see Phasing).

### Commands

```sh
cargo build --workspace
cargo test --workspace
cargo run -p zerocache-http    # requires ZEROCACHE_PROVIDER_API_KEY set
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

trait EmbeddingProvider: Send + Sync {
    fn embed_batch(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, ProviderError>;
}
```

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
zerocache-core            domain: CacheKey, reconciliation — no dependencies
zerocache-ports           trait definitions (incl. StoreError/ProviderError) — depends on core
zerocache-adapters-sled   EmbeddingStore impl, embedded/single-instance — depends on ports
zerocache-adapters-redis  EmbeddingStore impl, shared/multi-instance — depends on ports
zerocache-adapters-openai EmbeddingProvider impl — depends on ports
zerocache-http            axum transport, wire-shape translation, application wiring — depends on all of the above
```

`zerocache-adapters-redis` is not in the PRD §13 list — see "Deviations" below.

## Deviations from the PRD

The PRD (§3) is authoritative for intent, but treat these as the actual code, not what the literal PRD text says:

1. **`EmbeddingProvider::embed_batch` takes a `model: &str` parameter.** PRD §6.2's snippet omits it, but the wire contract (§10) requires honoring whatever `model` the client sends per-request — without it, one adapter instance could only ever serve a single hardcoded model.
2. **`EmbeddingStore`/`EmbeddingProvider` carry `Send + Sync` bounds**, needed because `zerocache-http` shares them across async axum handlers behind `Arc<dyn Trait>`.
3. **Both traits return `Result`, not bare values.** Originally scaffolded without this (matching PRD §6.2 literally) and it was a real gap — failures panicked instead of surfacing as HTTP errors. Fixed 2026-07-23.
4. **`zerocache-adapters-redis` exists in v1.** PRD §4 and §6.4 explicitly defer distributed/multi-instance storage to "Future" work. Added ahead of that schedule because Kubernetes multi-replica deployment is a real near-term requirement, not a hypothetical — confirmed with the PRD's author before building. `zerocache-adapters-sled` remains the default; Redis is opt-in via `ZEROCACHE_STORAGE_BACKEND=redis`.
5. **`GET /metrics` exists in v1**, ahead of PRD §11's full observability spec. Prometheus text format, three counters (`zerocache_cache_hits_total`, `zerocache_cache_misses_total`, `zerocache_provider_prompt_tokens_total`) — deliberately scoped to just what Phase 1's success criteria (§15: measured hit rate, measured tokens billed) need, not per-consumer tagging or latency-saved-vs-baseline, which need decisions not yet made. Chosen over a bespoke JSON endpoint specifically because with `zerocache-adapters-redis` multi-replica deployments, per-pod counters only mean something once something aggregates across pods — Prometheus's scrape-and-`sum()` model does that; a single pod's JSON response can't.

## API contract (v1)

Single endpoint, shape preserved exactly so existing OpenAI-compatible clients need only a `base_url` change:

```text
POST /v1/embeddings
{ "model": "...", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

Responses also carry `X-Zerocache-Hits` / `X-Zerocache-Misses` headers; `usage` reflects only tokens actually billed for this request's misses (0 for an all-hit batch — not a placeholder).

## Multi-consumer design (PRD §7)

- All consumers go through one running Zerocache process over HTTP — they never touch the store directly. Store concurrency is Zerocache's problem to solve once.
- Cache key is content-addressed (`model_id + model_version + text`), so unrelated consumers embedding identical text under the same model share a hit for free. This also means a text-normalization bug in one consumer's requests can pollute the shared key space for every other consumer.
- Namespacing and auth are explicitly deferred (not designed away) until a second real consumer needs isolation — v1 assumes a single global key space and a trusted network boundary.

## Non-goals for v1 (do not implement without a scope discussion)

- Live/conversational query embedding caching (deferred — low reuse rate, unproven).
- Semantic/fuzzy similarity matching — v1 is exact-match only.
- Vector quantization/compression — deferred until a real hit-rate number justifies it.
- Multi-provider failover — single provider only. (Multi-instance *storage* is no longer a non-goal — see "Deviations" above; `zerocache-adapters-redis` supports it.)
- Any framework-specific SDK/client package in TS or Python — if a consumer needs to install a Zerocache-specific package, the neutrality goal has failed.

## Testing strategy (PRD §12) — order matters

1. **Core**: unit tests, no I/O — key derivation and reconciliation verified in isolation first. Done (`zerocache-core`, 6 tests).
2. **Application**: tested against mock `EmbeddingStore`/`EmbeddingProvider` — verifies orchestration without real network/disk. Done (`zerocache-http/src/app.rs` `#[cfg(test)]` module, 6 tests: hit/miss splitting, ordering, store/provider failure propagation, metrics-only-recorded-on-success).
3. **Adapters**: integration tests against real sled and a stubbed provider. Done — `sled` roundtrip test, plus 3 `httpmock`-stubbed `zerocache-adapters-openai` tests (response reordered by index, HTTP error status, malformed body). No test against real Redis (would need a running instance); real-provider behavior additionally verified manually via the local smoke test.
4. **End-to-end, consumer 1**: real batch run through Argus (TS/Mastra) — produces Phase 1 hit-rate/cost numbers. **Not started** — Argus is a separate, large project; wiring it up is out of scope until Argus itself is ready. A manual local smoke test (real OpenAI key, miss then repeat-hit) substituted for this on 2026-07-23 and caught a real startup panic, fixed in commit `714944b`.
5. **End-to-end, consumer 2**: real batch run through a second, Python-based ingestion script (LangChain/LlamaIndex) — this is the test that actually validates the neutrality claim, not #4. Not started, blocked on #4.

## Phasing (PRD §16)

- **Phase 1 (current)**: workspace setup, core logic, port traits, sled + single-provider adapters, HTTP wire-up against Argus, first hit-rate/cost measurement.
- **Phase 1.5**: wire up a second, Python-based consumer against the same instance for a second measurement.
- **Phase 2**: scoped only after Phase 1/1.5 results are in — quantization, eviction, namespacing, additional adapters. Do not pre-build these.
- **Distribution** (Docker image, prebuilt binaries, npm/pip wrapper launchers, crates.io) is explicitly sequenced *after* Phase 1 and 1.5 validation — do not add packaging work before there's a proven hit-rate number behind it.

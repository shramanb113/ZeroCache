# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only `PRD.md` — there is no Cargo workspace, no source code, and no tooling set up yet. Treat `PRD.md` as the authoritative spec: read it in full before scaffolding anything. There are no build/lint/test commands to document because nothing has been implemented. Once the workspace described below exists, this file should be updated with the real commands (`cargo build`, `cargo test -p <crate>`, etc.) and any CI/lint setup actually added.

## What Zerocache is

A standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider (OpenAI-compatible endpoint). It intercepts `/v1/embeddings` requests, serves previously-computed vectors from a local content-addressed store, and forwards only cache misses upstream. Adoption requires no SDK — a consumer just points its existing OpenAI-compatible embedding client at Zerocache's `base_url`.

First validation consumer: Argus (Mastra/TS + pgvector). A second, Python-based consumer (LangChain/LlamaIndex) is required before the "any framework" neutrality claim is considered proven, not just asserted (PRD §5, §14, §15).

## Architecture — layered, dependencies point inward only

This is a hard, structurally-enforced rule (via Cargo workspace crate boundaries, not convention): outer layers know about inner layers; inner layers never know outer layers exist.

| Layer | Responsibility | Depends on | Knows about I/O? |
|---|---|---|---|
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
Trait contracts that core/application depend on but never implement directly:
```rust
trait EmbeddingStore {
    fn get(&self, key: &CacheKey) -> Option<Vec<f32>>;
    fn put(&self, key: CacheKey, vector: Vec<f32>);
}

trait EmbeddingProvider {
    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>>;
}
```

### Application
Runs the actual cache flow against the two ports: split batch into hits/misses, call the provider port for misses only, write results back through the store port, reassemble the final ordered response. Fully testable with mock ports — no real network/disk required. Also unaware of wire shape; that translation happens once, at the transport boundary.

### Adapters
Swappable concrete implementations, each independent:
- `EmbeddingStore`: v1 = sled-backed, raw fp32, no compression, no eviction (content-addressed keys mean a hit is valid indefinitely — eviction is a capacity concern only, never a correctness one).
- `EmbeddingProvider`: v1 = single OpenAI-compatible adapter. Future adapters (separate crates, no change to core/ports/application): Mistral, Gemini, Voyage AI, Cohere.

Adapters depend inward on port traits only — adding a new adapter never touches core or application.

### Interface/Transport
Two deliberately separate responsibilities:
- **Wire-shape translation**: parse the incoming request (v1: OpenAI `/v1/embeddings` shape — TS and Python frameworks alike default their embedding clients to it) into the transport-agnostic internal request format.
- **HTTP surface**: an axum server handling concurrent requests from multiple independent consumers.

Keeping these separate is what lets a second wire shape be added later as just a new translation function, invisible to application/core.

## Planned workspace structure (PRD §13)

```
zerocache-core            domain: CacheKey, reconciliation — no dependencies
zerocache-ports           trait definitions — depends on core
zerocache-adapters-sled   EmbeddingStore impl — depends on ports
zerocache-adapters-openai EmbeddingProvider impl — depends on ports
zerocache-http            axum transport, wire-shape translation, application wiring — depends on all of the above
```

## API contract (v1)

Single endpoint, shape preserved exactly so existing OpenAI-compatible clients need only a `base_url` change:
```
POST /v1/embeddings
{ "model": "...", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

## Multi-consumer design (PRD §7)

- All consumers go through one running Zerocache process over HTTP — they never touch the store directly. Store concurrency is Zerocache's problem to solve once.
- Cache key is content-addressed (`model_id + model_version + text`), so unrelated consumers embedding identical text under the same model share a hit for free. This also means a text-normalization bug in one consumer's requests can pollute the shared key space for every other consumer.
- Namespacing and auth are explicitly deferred (not designed away) until a second real consumer needs isolation — v1 assumes a single global key space and a trusted network boundary.

## Non-goals for v1 (do not implement without a scope discussion)

- Live/conversational query embedding caching (deferred — low reuse rate, unproven).
- Semantic/fuzzy similarity matching — v1 is exact-match only.
- Vector quantization/compression — deferred until a real hit-rate number justifies it.
- Multi-provider failover, distributed/multi-instance storage — single provider, single-node store.
- Any framework-specific SDK/client package in TS or Python — if a consumer needs to install a Zerocache-specific package, the neutrality goal has failed.

## Testing strategy (PRD §12) — order matters

1. **Core**: unit tests, no I/O — key derivation and reconciliation verified in isolation first.
2. **Application**: tested against mock `EmbeddingStore`/`EmbeddingProvider` — verifies orchestration without real network/disk.
3. **Adapters**: integration tests against real sled and a stubbed provider.
4. **End-to-end, consumer 1**: real batch run through Argus (TS/Mastra) — produces Phase 1 hit-rate/cost numbers.
5. **End-to-end, consumer 2**: real batch run through a second, Python-based ingestion script (LangChain/LlamaIndex) — this is the test that actually validates the neutrality claim, not #4.

## Phasing (PRD §16)

- **Phase 1 (current)**: workspace setup, core logic, port traits, sled + single-provider adapters, HTTP wire-up against Argus, first hit-rate/cost measurement.
- **Phase 1.5**: wire up a second, Python-based consumer against the same instance for a second measurement.
- **Phase 2**: scoped only after Phase 1/1.5 results are in — quantization, eviction, namespacing, additional adapters. Do not pre-build these.
- **Distribution** (Docker image, prebuilt binaries, npm/pip wrapper launchers, crates.io) is explicitly sequenced *after* Phase 1 and 1.5 validation — do not add packaging work before there's a proven hit-rate number behind it.

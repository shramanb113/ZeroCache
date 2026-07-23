# Zerocache — Product Requirements Document

**Status:** Draft v2 — scope widened from single-consumer to any TS/Python agent orchestration framework
**Owner:** Shraman
**First consumer:** Argus (Mastra/TS + pgvector RAG pipeline) — validation case, not the target user
**Target consumers:** any embedding-ingestion pipeline built on a TS or Python agent orchestration framework (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack, and equivalents)

---

## 1. Summary

Zerocache is a standalone, Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider. It intercepts embedding requests, serves previously-computed vectors from a local content-addressed store, and forwards only the cache misses upstream. It is API-compatible with existing embedding provider endpoints, so any orchestration framework — regardless of language — can adopt it by changing a base URL, with no framework-specific integration code and no SDK to install.

## 2. Problem Statement

RAG ingestion pipelines re-embed text that has already been embedded before — during re-indexing, pipeline re-runs, development iteration, or overlapping corpora across projects. Every re-embed is a wasted provider call: it costs input tokens and adds latency, and the result is byte-identical to something already computed. This is not specific to one framework or one language — it happens identically whether the pipeline is TS-based or Python-based. No caching layer exists today that is (a) provider-API-compatible out of the box, (b) implemented as a standalone service rather than a library wrapper tied to one ecosystem, and (c) genuinely validated across more than one language's tooling rather than assumed to work by virtue of "just being an HTTP service."

## 3. Goals

- Eliminate redundant embedding provider calls for text that has already been embedded under the same model and model version, regardless of which framework or language produced the request.
- Be adoptable by any TS or Python agent orchestration framework without code changes — compatibility is achieved at the wire level (an existing embeddings API shape), not via a client library in either ecosystem.
- Produce a real, measurable hit-rate and cost/latency delta on a live pipeline (Argus first), then repeat that measurement against a second, structurally different consumer to validate the neutrality claim rather than assume it.
- Keep the core cache logic decoupled from any specific storage engine, embedding provider, _or wire shape_, so all three can evolve independently of the cache logic itself.

## 4. Non-Goals (v1)

- Live/conversational query embedding caching. Reuse rates for unique user queries are low and the hit-rate story is unproven; this is explicitly deferred.
- Semantic/fuzzy similarity matching (returning a cached vector for a _similar but not identical_ input). v1 is exact-match only.
- Vector quantization and storage compression. Deferred until a real hit-rate number justifies the engineering cost.
- Multi-provider failover, distributed/multi-instance storage. Single provider, single-node store for v1.
- Any framework-specific SDK or client package, in either TS or Python. If a consumer needs to install a Zerocache-specific package to use it, the neutrality goal has failed.

## 5. Consumers

Argus (Mastra/TS + pgvector) is the first integration and the source of the Phase 1 measured result — its embedder config changes only its `base_url` to point at Zerocache. It is a validation case, not the design target: the architecture is not "Argus's cache," it is a general-purpose cache that Argus happens to be first in line to use. A second consumer on the Python side (e.g. a LangChain or LlamaIndex ingestion script) is required before the "any framework" claim in Section 3 can be considered proven rather than asserted — see Section 14.

## 6. System Architecture

Zerocache is structured in layers with a single rule governing all of them: **dependencies point inward only.** Outer layers know about inner layers; inner layers never know outer layers exist. This is enforced structurally via a Cargo workspace, not by convention.

| Layer               | Responsibility                                               | Depends on  | Knows about I/O?          |
| ------------------- | ------------------------------------------------------------ | ----------- | ------------------------- |
| Core (domain)       | Cache key derivation, hit/miss reconciliation logic          | nothing     | No                        |
| Ports               | Trait contracts for storage, provider, and wire-shape access | Core        | No (traits only)          |
| Application         | Orchestrates the end-to-end cache flow using ports           | Core, Ports | No (works through traits) |
| Adapters            | Concrete implementations of ports                            | Ports       | Yes                       |
| Interface/Transport | HTTP surface, wire-shape translation, response shaping       | Application | Yes                       |

### 6.1 Core (domain)

Pure logic. No async runtime, no network, no disk access, nothing imported from `tokio`, `axum`, or any storage crate.

Contains:

- `CacheKey` — derived as `blake3(model_id + model_version + text)`. Including model identity in the key is deliberate: it prevents a model or version upgrade from silently returning a stale vector that merely looks like a valid hit.
- Hit/miss reconciliation — given a batch of keys and a set of lookup results, determine which inputs are hits and which are misses, and preserve original input ordering through to the final response.

This layer has no awareness of which framework or language sent the request — that distinction is resolved entirely in the transport layer, before anything reaches core. This is what makes "any framework" an architectural property rather than a marketing claim.

### 6.2 Ports

Trait definitions that the core and application layers depend on, but do not implement:

```rust
trait EmbeddingStore {
    fn get(&self, key: &CacheKey) -> Option<Vec<f32>>;
    fn put(&self, key: CacheKey, vector: Vec<f32>);
}

trait EmbeddingProvider {
    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>>;
}
```

Application logic is written against these traits. It has no knowledge of sled, OpenAI, or any concrete implementation.

### 6.3 Application (orchestration)

Given the two ports above, this layer runs the actual cache flow: split a batch into hits and misses, call the provider port for misses only, write new results back through the store port, and reassemble the final ordered response. This layer is fully testable with mock ports — no real network or disk needed to verify correctness. It is also fully unaware of which wire shape the original request arrived in — that translation happens once, at the transport boundary, before this layer is ever invoked.

### 6.4 Adapters

Concrete implementations of the ports, each swappable independently:

- `EmbeddingStore`: v1 is a sled-backed store, raw fp32, no compression. Future: quantized (int8/fp16) store, distributed store (Redis) for multi-instance deployments.
- `EmbeddingProvider`: v1 is a single OpenAI-compatible adapter. Future adapters (Phase 2, added as separate crates, no change to core/ports/application): Mistral (`mistral-embed`), Google's Gemini embedding model, Voyage AI (the correct target for Claude-ecosystem consumers, since Anthropic does not offer its own embedding model), and optionally Cohere `embed-v4`. Multi-provider fallback is a further-out concern once more than one adapter exists.

Because adapters depend inward on the port traits and never the reverse, adding a new adapter never requires touching core or application logic.

### 6.5 Interface/Transport

The layer that actually carries the "any framework" claim, and the one place the multi-consumer scope genuinely changes the architecture versus a single-consumer design.

It has two responsibilities that are deliberately kept separate:

- **Wire-shape translation** — parsing an incoming request in whatever shape the calling framework expects (v1: the OpenAI `/v1/embeddings` shape, since TS and Python frameworks alike default to it — Mastra, LangChain, and LlamaIndex all support pointing an OpenAI-compatible client at a custom `base_url` with zero code change) and translating it into the transport-agnostic internal request the application layer understands.
- **Serving the HTTP surface itself** — an axum server handling concurrent requests from however many independent consumers are pointed at it.

Decoupling wire-shape parsing from the application layer is what lets a second wire shape (e.g. a Cohere-shaped endpoint, if a future consumer needs it) be added later as a new translation function, without the application or core layers ever noticing more than one shape exists.

## 7. Multi-Consumer Design

This is the section that changes materially now that Zerocache is not "Argus's cache":

- **Concurrency:** consumers never touch the store directly — every request goes through one running Zerocache process over HTTP. Store-level concurrency is therefore Zerocache's problem to solve once (via sled's internal thread-safety plus the async server), not something every consumer needs to reason about.
- **Cache sharing across consumers:** because the cache key is content-addressed (`model_id + model_version + text`), two unrelated consumers embedding the same text under the same model get a shared hit for free. This is a genuine benefit, not a side effect to design around — but it means a bug in one consumer's request formatting (e.g. inconsistent text normalization before hashing) can pollute the cache for every other consumer sharing that key space.
- **Namespacing (deferred, not designed away):** v1 ships with a single global key space, since the shared-hit benefit above is worth more than isolation at this stage and there's only one real consumer to validate against. An optional namespace prefix on the cache key (e.g. per API key or per project) is the natural extension once a second consumer actually needs isolation — not before.
- **Auth (deferred):** v1 assumes a trusted network boundary (local or same-VPC deployment). No per-consumer authentication is in scope until there's a concrete reason to need it.

## 8. End-to-End Data Flow

1. Client (Argus's Mastra embedder, or any other TS/Python framework configured the same way) sends a batch embeddings request to Zerocache's HTTP endpoint instead of the provider directly.
2. Zerocache's transport layer parses the request's wire shape and translates it into the internal batch format.
3. For each input text, the core layer derives `CacheKey = blake3(model_id, model_version, text)`.
4. The application layer performs a batch lookup against the `EmbeddingStore` port.
5. Reconciliation splits the batch into hits and misses, preserving original order.
6. Only the misses are forwarded, as a single batched call, to the real `EmbeddingProvider` adapter.
7. Returned vectors for the misses are written back to the `EmbeddingStore`.
8. The application layer reassembles the full response (hits + new) in original input order.
9. The transport layer translates the response back into the calling framework's expected wire shape.
10. Metrics are recorded: hit count, miss count, tokens avoided (estimated locally via tokenizer, since the provider only bills for misses), and cache-layer latency overhead — tagged by which consumer generated the request, for later hit-rate comparison across frameworks.

## 9. Storage Design

- **v1:** sled (pure Rust, embedded), storing raw fp32 vectors keyed by `CacheKey`. No eviction policy — because the key is content-addressed, a hit is valid indefinitely for that exact (model, version, text) triple; eviction is only a capacity concern, not a correctness one.
- **Deferred:** quantization (fp32 → int8/fp16 with a per-vector scale factor) to reduce storage footprint; LRU/capacity-based eviction once storage size becomes a real constraint; distributed backing store for multi-instance deployments; per-consumer namespacing (Section 7).

## 10. API Contract

v1 exposes a single endpoint matching the shape of a standard embeddings API, chosen specifically because both TS and Python framework ecosystems already default their embedding clients to it:

```
POST /v1/embeddings
{ "model": "...", "input": ["text1", "text2", ...] }

→
{ "object": "list", "data": [ { "embedding": [...], "index": 0 }, ... ], "model": "...", "usage": {...} }
```

The response shape is preserved exactly so that existing client libraries in either language require no code change beyond the base URL.

## 11. Observability & Metrics

Tracked internally (not visible in the provider's own dashboard, since it only sees the misses):

- Cache hit rate, per consumer and cumulative
- Estimated tokens avoided (computed locally from a tokenizer, not billed data)
- Latency added by the cache-check itself
- Latency saved versus an uncached run (cold-path baseline)

Per-consumer tagging exists specifically so that a hit-rate comparison between Argus (TS) and a second, Python-based consumer is a real number, not an assumption.

## 12. Testing Strategy

- **Core:** unit tests, no I/O — key derivation and reconciliation logic verified in isolation first.
- **Application:** tested against mock `EmbeddingStore`/`EmbeddingProvider` implementations — verifies orchestration correctness without real network or disk.
- **Adapters:** integration tests against real sled and a stubbed provider.
- **End-to-end, consumer 1:** a real batch run through Argus's ingestion pipeline (TS/Mastra), producing the Phase 1 hit-rate/cost numbers.
- **End-to-end, consumer 2:** a real batch run through a second, Python-based ingestion script (LangChain or LlamaIndex), producing a second measurement — this is the test that actually validates the neutrality claim, not the first one.

## 13. Workspace Structure

```
zerocache-core           (domain: CacheKey, reconciliation — no dependencies)
zerocache-ports          (trait definitions — depends on core)
zerocache-adapters-sled  (EmbeddingStore impl — depends on ports)
zerocache-adapters-openai (EmbeddingProvider impl — depends on ports)
zerocache-http           (axum transport, wire-shape translation, application wiring — depends on all of the above)
```

Cargo's crate boundaries make the dependency direction a compile-time guarantee rather than a convention that erodes under deadline pressure.

## 14. Risks & Mitigations

| Risk                                                                                 | Mitigation                                                                                                        |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Model/version upgrade silently returns stale vectors                                 | Cache key includes `model_id` + `model_version`, not text alone                                                   |
| Reconciliation bug (misordered or misclassified hit/miss) invisible until production | Core layer unit-tested in isolation before any adapter touches real I/O                                           |
| Building storage optimization before proving value                                   | Quantization and eviction explicitly deferred past v1                                                             |
| Applying this to live queries instead of static ingestion collapses hit rate         | Non-goal explicitly scoped out for v1; ingestion-only                                                             |
| "Any framework" claim proven only against one TS consumer, not actually validated    | Section 5/12 require a second, Python-based consumer before the claim is treated as true                          |
| Shared global key space lets one consumer's bug pollute another's cache              | Accepted trade-off for v1; namespacing designed but deliberately deferred until a second consumer needs isolation |

## 15. Success Criteria

Phase 1 is complete only when both of these are true, not one:

1. A real run of Argus's ingestion pipeline (TS/Mastra) through Zerocache produces a measured hit rate, a measured reduction in provider-billed tokens, and a measured latency delta versus the uncached baseline.
2. A real run of a second, Python-based ingestion pipeline through the same running Zerocache instance produces its own measured numbers, without any code change to Zerocache itself.

Anything short of both is a single-consumer tool with a neutrality claim attached, not a neutral cache.

## 16. Roadmap / Phasing

**Phase 1 (current scope):** workspace setup, core logic, port traits, sled + single-provider adapters, HTTP wire-up against Argus, first real hit-rate/cost measurement.

**Phase 1.5:** wire up a second, Python-based consumer against the same instance and produce a second measurement — this is what upgrades "framework-neutral" from a design intent to a proven property.

**Phase 2:** to be scoped once Phase 1 and 1.5's measured results are in hand — quantization, eviction, namespacing, additional adapters, and any surface expansion will be prioritized against what the real numbers show, not decided in advance.

## 17. Distribution (post-validation, not a Phase 1 task)

Deliberately sequenced after Phase 1 and 1.5 — packaging a cache with no proven hit-rate number is polish with nothing behind it.

- **Docker image** (GHCR/Docker Hub) — the primary distribution surface. Self-contained, works identically whether the consumer is TS or Python, no toolchain required on their end.
- **Prebuilt binaries** (GitHub Releases, cross-compiled for linux-x64/arm64, macOS, Windows) — for consumers who don't want to run Docker.
- **npm and pip wrapper packages** — thin postinstall launchers that fetch the correct platform binary and exec it (the same pattern esbuild/swc/biome use), so a TS consumer runs `npx zerocache` and a Python consumer runs `pip install zerocache && zerocache` without either installing an SDK. This is the distribution-layer equivalent of the OpenAI-shaped endpoint: meet each ecosystem in its own package manager instead of asking it to adopt a new tool.
- **crates.io** — lowest priority; for the rare consumer embedding Zerocache as a Rust library directly rather than running it as a service.
- **Configuration surface** across all of the above: environment variables only (provider API key, storage path, port, provider base URL) so behavior is identical whether launched via binary, Docker, or an npm/pip wrapper.

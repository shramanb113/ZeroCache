# Decisions

A running log of significant architectural decisions for Zerocache, with the reasoning behind them. Append new entries as decisions are made — don't rewrite history when a decision is later reversed; add a new entry and cross-reference the old one.

---

## 2026-07-24: Multi-tenant, multi-provider embedding proxy

**Decision:** Zerocache moves from "cache in front of one configured OpenAI-compatible provider" to "cache in front of any embedding provider, with each caller bringing their own API key." Full design spec lives at `docs/superpowers/specs/2026-07-24-multi-tenant-multi-provider-design.md` (gitignored — local working document, not published).

### Provider selection: path-based routing

Endpoint becomes `POST /{provider}/v1/embeddings` (e.g. `/openai/v1/embeddings`, `/mistral/v1/embeddings`) instead of a single `/v1/embeddings`.

**Why:** every major agent orchestrator (Mastra, LangChain, LlamaIndex, LangGraph, CrewAI, Haystack) wraps the official OpenAI SDK client, which exposes exactly three configurable knobs for a custom endpoint: `base_url`, `api_key`, `model`. None of them reliably expose a way to inject a custom JSON body field into the request, which rules out an explicit `provider` field in the body. Provider-in-path needs only `base_url` — the one knob already required for the "any framework, zero SDK" premise to work at all.

**Alternatives rejected:**
- Explicit `provider` field in the request body — cleanest on paper, but most orchestrator SDKs build the request body from a fixed typed parameter set with no escape hatch for extra fields.
- Prefix in the `model` field (`"mistral/mistral-embed"`, OpenRouter-style) — nearly as broadly compatible, but risks colliding with downstream tooling that does its own model-name lookups (pricing tables, tokenizer selection) expecting an unprefixed name.

### Tenant identity: hash of the forwarded API key

`owner_id = blake3(raw forwarded API key)`. The raw key is never stored or logged — only its hash.

**Why:** with no real auth system in place, and a cache hit never touching the upstream provider (so nothing ever verifies a claimed identity), the real question is which identity signal is hardest to guess or impersonate. A real provider API key is a long, high-entropy secret. An explicit, caller-chosen tenant header (e.g. `X-Zerocache-Owner: acme-corp`) would be low-entropy and unverified — anyone could type in someone else's tenant name and read their cached vectors for free.

**Trade-off accepted:** rotating an API key changes `owner_id`, so a tenant's cache goes cold on rotation. This is an operational inconvenience, not a security flaw — recorded here explicitly so it isn't discovered as a surprise later.

### BYOK fully replaces static provider configuration

`ZEROCACHE_PROVIDER_API_KEY` and `ZEROCACHE_PROVIDER_BASE_URL` are removed. Zerocache holds zero provider credentials of its own; every request must carry `Authorization: Bearer <key>`, with no fallback — including for what would otherwise be a cache hit, since `owner_id` can't be derived without a key.

### Cache key gains `provider` and `owner_id`

New derivation: `blake3(owner_id ++ provider ++ model ++ model_version ++ text)`. `provider` prevents two different providers' identically-named models from ever colliding; `owner_id` scopes the cache per-tenant so one caller's data or spend is never silently shared with an unrelated caller.

### Adapter scope for this pass: OpenAI, Mistral, Gemini

Three real `EmbeddingProvider` implementations — enough to prove the adapter pattern generalizes across genuinely different wire shapes and auth schemes (Gemini needs `x-goog-api-key`, not a bearer token), without pre-building every provider speculatively (YAGNI, matches PRD §16's phasing philosophy).

Any provider that's already OpenAI-wire-compatible (a self-hosted vLLM/Ollama server, potentially OpenRouter if it turns out to expose embeddings) needs zero new adapter code — just a new `base_url` pointed at the existing OpenAI adapter. A registration mechanism for arbitrary custom endpoints (a config file/list) is deliberately deferred; three known, hardcoded providers don't yet justify building a config surface for an open-ended list.

### Metrics get a `provider` label, never an `owner` label

`zerocache_cache_hits_total{provider="mistral"}` etc. Provider is a small, bounded, non-sensitive label — and this is incidentally the "per-consumer tagging" PRD §11 deferred, now available for free. Owner/tenant is explicitly excluded from metrics: it would leak tenant identity into a monitoring system and create one time series per tenant (unbounded cardinality).

---

## 2026-07-24: Real second-consumer testing found a real wire-contract bug mocks couldn't

**What happened:** Built a real LangChain TypeScript RAG demo (`demo/langchain-ts/`) specifically to battle-test Zerocache from an unmodified real client's perspective, not just its own mocks. It found, within the first real-key run, that `embedQuery()` — LangChain's standard single-text embedding call, not an edge case — failed 100% of the time against Zerocache with a `422`.

**Root cause:** `EmbeddingsRequest.input` in `zerocache-http/src/wire.rs` required a JSON array (`Vec<String>`). OpenAI's real `/v1/embeddings` endpoint accepts `input` as either a single string or an array of strings; `embedQuery()` sends the bare-string form, since that's the natural, documented shape for a single piece of text. Every existing Zerocache test — unit, integration, and the httpmock-stubbed provider tests — constructed `input` as an array, because that's what the plan/brief code and every hand-written test happened to use, not because anyone verified it against a real, unmodified client's actual wire behavior.

**Why mocks and existing tests couldn't have caught this:** the bug is entirely in what shape of JSON Zerocache's *own* deserializer accepts on the way in — no mock of the store or provider is anywhere near this code path. It only surfaces when something outside the codebase's control (a real client library) sends a request shaped differently than every internal test assumed. This is the concrete argument for PRD §12's staged testing plan insisting on a real second consumer, not settling for #4's manual smoke test alone.

**Real-world severity:** this didn't just fail an isolated call — `MemoryVectorStore.similaritySearch()` calls `embedQuery()` internally for the query text, so *every* RAG question-answering call failed, while document ingestion (which always batches, using the array form) worked fine. A demo — or a real production RAG app — would have looked completely functional through ingestion and then failed on literally the first real user question.

**Fix:** `EmbeddingsRequest.input` now uses a custom `deserialize_with` (an untagged `String | Vec<String>` enum, wrapping a single string to a one-element `Vec`) so both shapes are accepted, matching OpenAI's actual contract. `DeleteRequest` shares the same struct, so it's fixed too, for free.

**Also measured, not just reasoned about:** the same testing pass fired 5 concurrent requests for one never-before-seen text through Zerocache. All 5 independently missed the cache and called the provider — 4 of 5 calls were purely redundant spend. This is the first real, measured number behind the request-coalescing gap that's been discussed and deferred ("singleflight") across several sessions; previously it was only a theoretical concern.

---

## 2026-07-24: In-process request coalescing, scoped deliberately narrow

**Decision:** implement singleflight-style coalescing for concurrent identical-key misses within one Zerocache instance, immediately after the above finding gave it a real number (4 of 5 concurrent calls being pure waste). Full mechanics in `CLAUDE.md` Deviations item 11.

**What's coalesced vs. what isn't, and why:** only the provider call itself. The store write stays un-deduped — every request that resolves a key, whether it claimed the fetch or piggybacked on someone else's, still writes its own copy back. This was a deliberate simplification, not a missed case: this codebase already treats redundant writes of an identical value as correctness-neutral (it's the exact argument `zerocache-adapters-redis`'s own doc comment makes for why concurrent replicas racing to fill the same key is safe — last-write-wins is fine when the value can only ever be one thing). Coalescing the cheap half of the operation would have added real complexity (a second shared-future type, or bundling the write into the same detached task) for a cost that was already negligible. Coalescing the expensive half — the actual billed provider call — is where all the real value is.

**Why in-process, not cross-replica:** the `in_flight` map lives in `AppState`, one per process. Two Zerocache replicas behind a load balancer each run their own map, so the same concurrent-miss pattern can still cost 2 provider calls instead of 1 across a multi-replica deployment, even after this fix. Solving that needs a distributed coordination primitive — a Redis-backed lock or lease, with its own failure modes (what happens if the lock holder dies mid-fetch) — genuinely harder work than the in-process version, and deliberately not attempted in the same pass. This was flagged as the right sequencing before any code was written: build the easy, high-value version first, scope the hard version separately rather than let it block or bloat the first one.

**Why usage attribution needed its own rule:** a coalesced request's response still needs to report token usage somehow, and the naive answer (every coalesced sibling reports the full batch's tokens) would silently inflate `zerocache_provider_prompt_tokens_total` — a metric this project already treats as a source of truth for "tokens actually billed" (see the BYOK deviation entry above). The rule adopted: only the request that actually triggered the provider call attributes tokens to itself; every piggybacking request reports zero for the keys it coalesced on, identical to how a cache hit already reports zero. This keeps the metric's meaning intact without needing to fractionally split one API call's token count across N requests that share it.

---

## 2026-07-28: A generalized cloud adapter layer, and the cache-key break it required

**Decision:** add real `EmbeddingProvider` adapters for Azure OpenAI/Foundry Models, Amazon Bedrock, and GCP Vertex AI — the work Deviations item 17 explicitly deferred — behind a shared kit crate, `zerocache-adapters-cloud`, rather than three independent adapters copying the existing four-adapter pattern. Full mechanics in `CLAUDE.md` Deviations items 18 and 19; this entry is the reasoning behind the choices those items describe.

**Why a shared kit crate, and why that isn't a violation of "dependencies point inward only":** the inward-dependency rule is about direction, not about forbidding shared code — it says outer layers may know about inner layers and never the reverse, not that every adapter must be a self-contained island. Azure, Bedrock, and Vertex AI each front *multiple independent vendors* behind one cloud API (Azure serves both OpenAI models and Foundry-hosted Cohere; Bedrock serves both Titan and Cohere), so three adapters written from scratch would have triplicated the same client-construction, timeout, retry, chunking, count-checking, and usage-accumulation logic the four existing adapters already each carry once. `zerocache-adapters-cloud` sits between `zerocache-ports` and the three cloud crates in the dependency graph — it depends on ports, and each cloud crate depends on it, exactly one layer removed from where a direct-to-ports adapter would sit. No adapter depends on another adapter; the rule holds. The four existing wire-shape adapters were deliberately left un-migrated onto the kit — doing so would have been a large, purely-internal diff across four already-working, already-tested crates for no user-visible benefit, and would have dragged them into this change's review and risk surface for nothing this feature needed.

**Why no AWS SigV4 for Bedrock:** the obvious "correct" AWS integration is SigV4-signed requests using an access-key/secret pair and a region. That was rejected because Zerocache is a BYOK proxy — the caller's credential travels in one `Authorization` header, the same shape every other provider already uses. SigV4 needs an access key, a secret key, and a region as three separate values, which would mean either inventing a new multi-field wire contract (a header format or JSON body field nothing else in this codebase has) or asking callers to encode three values into one string and trusting Zerocache to parse it apart — a new wire contract either way, not an implementation detail hidden inside the adapter. AWS's `bedrock-runtime` ships first-class bearer API keys as a real, documented alternative to SigV4, which lets Bedrock fit the existing `Authorization: Bearer <key>` contract every other provider already uses, with zero new wire surface. This was the deciding factor, not general SigV4-avoidance — if Bedrock had no bearer-key option, SigV4 (and the wire-contract change it implies) would have been back on the table as a real, separate decision.

**Why cloud coordinates live inside the `model` string:** the same reasoning the 2026-07-24 provider-selection decision already established for provider-in-path applies again here — orchestrator SDKs expose `base_url`/`api_key`/`model` and nothing else reliably injectable, so a new header or body field for region/project/location would be invisible to most real callers. `model` is already free-form per-request, already folds into the cache key, and is settable from any framework without a Zerocache-specific SDK. The accepted consequence: a malformed `model` string (bad region, unparseable grammar) surfaces today as a `502`, not a `400`, because every `AppError::Provider` currently maps to `502` regardless of whether the underlying `ProviderError` is "upstream is having a bad day" or "you gave me garbage I could never route." Splitting `ProviderError` into a client-fault and an upstream-fault variant (and giving each its own HTTP status) is a real, separate change with its own blast radius across every existing adapter's error handling — not bundled into this pass.

**The one-time cache invalidation, and why a compatibility shim was rejected:** adding `cache_scope` to `CacheKey::derive`/`derive_image` means every cache entry written before this change is unreachable after it — not corrupt, just permanently a miss, since content-addressed entries are never wrong, only absent. A shim that computed the old, `cache_scope`-less key as a fallback lookup on miss was considered and rejected: it would have reintroduced exactly the hazard `cache_scope` exists to close, since the whole point is that two genuinely different upstreams (two Bedrock regions, two Vertex projects) can share an old-format key and must never be treated as interchangeable. A shim can't tell "safe to fall back" apart from "silently wrong" without already having the information `cache_scope` was added to capture — so half-fixing it would have been strictly worse than a clean break. A one-time cold start across every deployment was judged the correct price against that alternative.

**The unresolved Vertex token-rotation risk:** `owner_id` is a hash of the caller's forwarded credential, and a GCP OAuth2 access token typically lives about an hour, so a Vertex caller's owner-scoped cache namespace rotates every time their token does — fine for a single long-running ingestion job, broken across multiple days of reuse. Two candidate fixes were identified and neither was taken: (1) a caller-supplied stable tenant header, which reintroduces the exact unverified-identity spoofing risk the 2026-07-24 entry above rejected for the general multi-tenant design — anyone could claim anyone else's tenant string; (2) deriving `owner_id` from a stable claim embedded inside the token itself (e.g. the service-account email) rather than hashing the token's raw bytes, which stays spoofing-resistant but requires parsing and trusting token internals per cloud, a real scope increase this pass didn't take on. Neither was picked unilaterally, because either one trades away a security property this file has already argued for on the record — that decision needs the PRD's author, not an agent, so it's recorded here as open rather than resolved.

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

---

## 2026-08-29: Chat-provider configuration surface

**Decision:** generalize the completion cache's single hardcoded `openai` chat provider (Deviations item 21) into a registry — a batteries-included built-in table of nine OpenAI-wire endpoints merged with an optional `ZEROCACHE_CHAT_PROVIDERS="name=url,…"` override var. Full mechanics in `CLAUDE.md` Deviations item 22.

**One map var, not per-provider `ZEROCACHE_<NAME>_CHAT_URL` vars:** per-provider vars only work for a fixed known set — adding `groq`/`deepseek`/a self-hosted box would each need a code change to read its var. The map handles any OpenAI-wire provider with zero code change, and the built-in table means the common providers need no config at all.

**The configured value is the URL prefix up to `/chat/completions`, not a bare origin:** the embedding path's `ZEROCACHE_*_BASE_URL` rule is "bare origin, adapter adds `/v1`". That cannot express Gemini's OpenAI-compat prefix `…/v1beta/openai`, which is exactly the doubled-`/v1` footgun item 21 recorded. The explicit-prefix rule has one form and no heuristic, and `normalize_chat_url` accepts the bare prefix, a trailing slash, or a pasted full completions URL; the startup log echoes the resolved URL per provider so a typo shows up at boot.

**Chat config is decoupled from `ZEROCACHE_OPENAI_BASE_URL`:** that var stays embeddings-only (its item-17 meaning). Coupling chat to it would either reintroduce the doubled-`/v1` footgun or force a heuristic join. The cost — a box serving both wire-compatible embeddings and chat now sets two vars — is worth the single clean mental model ("embeddings: `ZEROCACHE_*_BASE_URL`; chat: `ZEROCACHE_CHAT_PROVIDERS`").

## 2026-08-29: Semantic completion cache (opt-in, single-instance)

**Decision:** add a semantic near-match tier to the completion cache, gated behind a `--features semantic` compile flag *and* `ZEROCACHE_SEMANTIC=1` at runtime, sled-only. Full mechanics in `CLAUDE.md` Deviations item 24.

**candle, not ONNX Runtime:** an embedding model needs a runtime. ONNX Runtime is a C++ dependency with its own shared library — it would end the `FROM scratch` + static-musl image story for the default build and add a real build-toolchain requirement. candle is pure Rust, links statically, and the BERT model definition is ~200 lines in `candle-transformers`. The cost is candle compiling ~1 minute the first time and pulling `gemm`; that only happens in a `--features semantic` build, never in `cargo build --workspace`.

**Model compiled in, not fetched at runtime:** CLAUDE.md's "can't rot" principle rules out a startup download — a cache proxy that fails to boot because Hugging Face is down, or silently embeds with last week's model, is worse than one that's a few MB bigger. `all-MiniLM-L6-v2` f16 safetensors (~45 MB) is committed and `include_bytes!`d. The `:semantic` image lands ~60 MB vs the default ~14.7 MB; that tag is opt-in, the default is untouched.

**f16, not int8/GGUF, for v1:** candle's turnkey path for BERT is f32/f16; int8 quantized BERT needs more plumbing. f16 halves the committed blob vs f32 with negligible cosine error at a 0.97 threshold. int8/GGUF (~23 MB) is a documented later optimization, not a v1 requirement.

**A hard coarse-key-hash gate, not soft per-field re-verification:** when HNSW returns a near neighbour, the tier accepts it only if a blake3 hash of the *entire request with the fuzzy span blanked* matches byte-for-byte — same system prompt, same tools, same generation params, same prior turns — AND cosine ≥ threshold. The alternative (re-check a handful of fields individually) is an allowlist that silently misses any field it wasn't taught about; the hash is a denylist-by-construction. This makes the tier provably "exact match, minus the configured fuzzy span is fuzzy" rather than "approximately the same request." The match-unit discriminant is folded into the hash so changing `ZEROCACHE_SEMANTIC_MATCH_UNIT` can't produce a cross-unit false hit.

**sled only; redis warns and runs exact-only:** the index is rebuilt in memory from a `load_all()` enumeration at startup, which needs a store surface that iterates — a property the existing key-addressed store traits deliberately lack, so it's a new `CompletionVectorStore` port. Implementing it on redis, plus cross-replica index propagation (a Redis stream tailed into each replica's HNSW, or periodic reload), is real separate work — roadmap Project 3. Until then a redis deployment that sets `ZEROCACHE_SEMANTIC=1` gets a `warn` and exact-match caching, not a boot failure.

**Fail-fast on embedder-load failure when the tier is enabled:** if `ZEROCACHE_SEMANTIC=1` and `TextEmbedder::load()` errors, the process exits non-zero rather than starting without the tier. An operator who set that var asked for the tier; a silent downgrade to exact-only would hide a broken deploy behind a normal-looking startup.

**Threshold is the safety knob, not the embedder:** false positives (a wrong completion served as a hit) are bounded by the cosine threshold, not by model quality — a better embedder tightens the paraphrase cluster but the threshold is what says "close enough." Default `0.97`, floored at `0.5`, env-tunable; the paraphrase smoke test is the first real calibration point.

---

## 2026-08-29: Multi-replica semantic index — Redis Stream change-feed

**Decision:** The semantic completion tier (2026-08-29 entry above; CLAUDE.md Deviations item 24) becomes available on `ZEROCACHE_STORAGE_BACKEND=redis` across N replicas, backed by a single Redis Stream that is both the shared record log and the change-feed. Per-replica in-memory HNSW is unchanged; each replica tails the stream and applies remote inserts/deletes to its own graph. Full design: `docs/superpowers/specs/2026-08-29-multi-replica-semantic-index-design.md` (gitignored). Supersedes the "sled only; redis warns and runs exact-only" decision above for the redis case — the `warn`-and-disable branch is removed.

**Vanilla Redis, not Redis Stack / RediSearch:** native `FT.SEARCH ... KNN` would remove the per-replica index and all propagation code, but it forces the operator onto the RediSearch module instead of the plain `redis` image the project ships in `docker-compose` and assumes everywhere, and it adds a network round-trip to every semantic probe. Rejected for infra overhead. Recorded as the revisit path if stream memory ever becomes a real constraint.

**One Redis Stream as source of truth, not a separate authoritative key set:** `insert` -> `XADD` the full `VectorRecord`; `delete` -> `XADD` a tombstone; `load_all` / `changes_since` -> `XRANGE` folded by `exact_key`, last op wins. One structure, one code path, and the resume cursor falls out of the stream ID with no separate high-water query and no snapshot-vs-live race. A HASH-plus-signal-stream alternative reclaims memory on delete and avoids a resurrection edge, but at two data structures, two writes per insert, and an `HGETALL`/`HSCAN` boot path — not worth it given completion-vector deletes only happen on the rare self-heal path.

**Poll, not push:** a single background task per process calls `changes_since(cursor)` on a ~2 s interval (`ZEROCACHE_SEMANTIC_POLL_MS`, clamped `[250, 60000]`). No adapter-side subscriber thread, no `Condvar` — simpler than "Project 2"'s coordinator. The lag is benign: a not-yet-propagated vector is a semantic miss that falls through to the exact fetch, and "absent, never wrong" already governs the whole cache. Pub/sub or `XREAD BLOCK` to cut the lag is deferred until a measured workload needs it.

**Stream bounded by `MAXLEN ~` always, `MINID` when `ZEROCACHE_TTL_SECONDS` is set:** approximate size cap (`ZEROCACHE_SEMANTIC_INDEX_MAXLEN`, default 100k ~= 210 MB) on every `XADD`; default millisecond stream IDs let an opportunistic `XTRIM MINID` give the feed the same time-expiry the completion blob store has. A trimmed-but-still-live vector just becomes an exact-fetch miss on replicas that started after the trim — degradation, not corruption. The one edge that is a stale *hit* rather than a miss: if a `put` survives trimming but its later `del` does not, a replica that boots after the trim re-adds the vector and can serve one stale semantic hit until `semantic_probe`’s self-heal sees the missing completion blob and re-tombstones it — bounded, and within the “absent, never wrong” contract the cache already lives by.

**No new opt-in flag:** multi-replica behaviour is implied by `ZEROCACHE_SEMANTIC=1` + the redis backend. Independent of `ZEROCACHE_CROSS_REPLICA_COALESCING` ("Project 2") — the two compose but neither reads the other's state.

---

## 2026-08-30: Cross-replica request coalescing (opt-in, single-key)

Roadmap Project 2. Full design: `docs/superpowers/specs/2026-08-29-cross-replica-request-coalescing-design.md`. Five knobs, all settled with the user:

- **Single-key only.** Cross-replica coalesce only requests resolving to one `CacheKey` (all completions; single-`input` embeddings). A multi-key embedding batch would need K coordinated locks with deadlock-ordering and partial-fill bookkeeping, for a case the in-process dedup + per-replica in-flight map already largely cover. Deferred, not rejected.
- **Hybrid wait (pub/sub + poll).** Redis `PUBLISH`/`psubscribe` for a sub-millisecond wake, a 250 ms store re-read as the backstop for a missed publish (the SET-NX/subscribe race). Poll-only was the simpler fallback; the user chose the hybrid for wake latency.
- **Explicit opt-in.** `ZEROCACHE_CROSS_REPLICA_COALESCING=1`, redis backend only. Not auto-on-with-redis: a distributed lock in the hot path is a behaviour change an operator should choose. Sled + flag = startup warn + no-op.
- **Fixed 60 s lock TTL + explicit release, no watchdog.** Leader `DEL`s the lock (via a `GET`-guarded Lua script so it never deletes a successor's lock) on success and on failure. The full TTL only elapses on a leader crash. A renewal/watchdog task was rejected as complexity for a rare case.
- **Leader failure -> followers fall back to their own call.** No negative marker, no error caching -- preserves "never cache a non-2xx". The leader still `complete`s (releases + signals) on a non-2xx so peers stop waiting; they then poll, see no fill, and fall back. One follower may re-contend and promote.

Any Redis error in the coordinator degrades to the safe default (lead / wait-elapsed / ignore), so a coordination outage is exactly today's per-replica behaviour, never a failed request. `zerocache-adapters-redis` took no new dependency: `std` threads + the `redis` crate's sync pub/sub.

Independent of `ZEROCACHE_SEMANTIC` and the multi-replica semantic index (2026-08-29 entries above): the two features compose without interaction. `complete()` resolves a request exact-match -> semantic near-match -> provider call, and `coalesce_cross_replica` wraps only that final provider call, so a semantic hit returns early and never enters the cross-replica path.

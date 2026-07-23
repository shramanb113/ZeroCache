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

# Zerocache agent showcase

A bespoke multi-agent coding team ships a real change to a real repo, then
re-runs the **identical** task served entirely from Zerocache's cache:
byte-identical result, ~$0, a few seconds instead of a few minutes. A third
run with a **reworded** task still hits, through the semantic near-match tier.

This is the demo that answers "is this a fad?" — the agents do genuine
cross-file work and `node --test` goes red → green on screen; the cache just
makes every pass after the first one free.

The narration script for the video is in [`SCRIPT.md`](./SCRIPT.md).

## The problem this is about

Every team shipping AI features re-pays for work it already did:

- **CI / eval loops** re-run the same deterministic prompts hundreds of times a week.
- **Prompt iteration** changes one line and re-runs the whole agent.
- **Multi-agent systems** fan out near-identical templated prompts across workers.
- **Retry / flap storms** re-bill on every attempt.
- **Local dev** re-runs the same flows all day.

The second, third, hundredth time you run a deterministic LLM workload you pay
full freight again — tokens *and* wall-clock latency on the dev loop.

### What the ecosystem offers today, and why it isn't enough

| Option | Why it doesn't close the gap |
| --- | --- |
| **Provider-side prompt caching** (OpenAI auto, Anthropic `cache_control`) | Discounts *input* tokens only, still runs the request, still bills output, evicts in 5–60 min, per-provider, no cross-run reuse, no metrics you own. |
| **Framework caches** (LangChain `LLMCache`, LlamaIndex) | In-process, single-language, exact-match only, no multi-tenancy, no streaming, die with the process. |
| **Semantic-cache SaaS** (Portkey, Helicone, hosted GPTCache) | Your prompts and completions leave your infrastructure into a vendor DB — a non-starter for most enterprise deals — plus another bill, another dependency, latency on the hot path. |

**The gap:** nobody ships a provider-neutral, self-hosted, BYOK, exact **+**
semantic completion/embedding cache as one static binary you put in front of
any OpenAI- or Anthropic-shaped endpoint with a `base_url` change and zero SDK.
That is Zerocache.

## What each run proves

| Run | Command | What you see |
| --- | --- | --- |
| **1 — cold** | `npm run -- run --run=1` | Real API calls. The team plans, three coders work in parallel, Claude reviews (streaming), a fixer applies feedback, tests pass. The ledger shows real upstream calls, tokens, and dollars. |
| **2 — warm** | `npm run -- run --run=2` | The same task. Every call returns `X-Zerocache-*-Hit: true`, the review stream replays instantly, the diffs are byte-identical, tests pass. Wall time drops to seconds, cost to `$0.00`. |
| **3 — semantic** | `npm run -- run --run=3` | The task, **reworded**. The architect/fixer calls still hit — `X-Zerocache-Completion-Hit-Kind: semantic`, with the cosine score in a header — because the final user message is within threshold of run 1's. Needs a `--features semantic` Zerocache with `ZEROCACHE_SEMANTIC=1`. |

`npm run -- run --run=2 --check` asserts the warm run had **zero** cache
misses and produced a byte-identical working tree; it exits non-zero
otherwise. `npm run -- run --record` runs 1 → 2 → 3 and prints the full
savings report.

## Zerocache surfaces exercised

| Stage | Agent | Route | Surface |
| --- | --- | --- | --- |
| Retrieve | Architect | `POST /openai/v1/embeddings` | embeddings cache |
| Plan | Architect | `POST /openai/v1/chat/completions` | chat completions cache, multi-provider |
| Repo brief | 3× Coder (concurrent) | one **identical** `POST /openai/v1/chat/completions` fired concurrently | **in-process coalescing** (3 misses → 1 upstream call) |
| Implement | Coder ×4 files | `POST /openai/v1/chat/completions` | chat completions cache |
| Review | Reviewer | `POST /anthropic/v1/messages` (streaming) | **native Anthropic messages cache**, **streaming buffer-and-replay** |
| Fix | Fixer | `POST /openai/v1/chat/completions` | chat completions cache |
| _Retrieve (optional)_ | Architect | `POST /gemini/v1/images/embeddings` | image-embeddings cache — only when `GEMINI_API_KEY` is set |

Every request is BYOK — `Authorization: Bearer <that provider's key>`, never
a Zerocache key; the keys are never stored, only hashed for per-caller cache
isolation. `temperature: 0` throughout (also what the cache gate requires).

## Why the cache is safe to trust

- **Content-addressed.** The key is `blake3(owner + provider + cache_scope +
  model + adapter_version + canonicalized_request)`. A hit is only ever a hit
  for a byte-identical deterministic request from the same caller.
- **Model identity is in the key.** Change the model, or upgrade the adapter,
  and every entry for the old identity is simply unreachable — a stale-but-
  plausible completion can't be served after a model bump.
- **Only 2xx, deterministic responses are stored.** A non-2xx upstream is
  forwarded with its real status and never cached.
- **Absent, never wrong.** A record that no longer deserializes, or a semantic
  neighbour whose blob has TTL-expired, is treated as a miss, not an error.

## Prerequisites

- **`OPENAI_API_KEY`** — required (coders, architect, fixer, text embeddings).
- `ANTHROPIC_API_KEY` — recommended. Without it the reviewer falls back to an
  OpenAI-wire model and the header shows `degraded: anthropic→openai`.
- `GEMINI_API_KEY` — optional, off by default. Set it (and drop a real PNG at
  `target-repo/docs/architecture.png`) to also exercise the image-embeddings
  cache.
- A running Zerocache:
  ```sh
  cargo run -p zerocache-http                                             # runs 1 and 2
  ZEROCACHE_SEMANTIC=1 cargo run -p zerocache-http --features semantic    # also run 3
  ```
- Node ≥ 22.

## Run it

```sh
cd demo/agent-showcase
npm install
cp .env.example .env          # fill in the keys you have

npm run warm                  # pre-fills the cache (runs 1 → 2 → [3])
npm run -- run --run=2        # the hero shot: identical task, fully cached
npm run -- run --run=3        # the reworded task, semantic hits
```

## Where the numbers come from

Every run appends a structured trace to `traces/run-{cold,warm,semantic}.jsonl`.
`src/trace.ts` (`summarize` / `compare`) turns those into the ledger and the
savings report. The committed trace files are from a real pass — the README,
the website, and `SCRIPT.md` quote them directly, nothing is hand-typed.

## Keyless smoke

`npm run smoke` starts `scripts/stub-zerocache.mjs` (a fake Zerocache that
needs no keys and no real instance) and runs the full pipeline against it —
useful for checking the rendering and the orchestration end-to-end in CI.
The stub's canned responses are not representative of real model output;
it only proves the plumbing.

## Layout

```
run.ts / warm.ts        CLI entry + cache pre-warm
src/orchestrator.ts     the agent team (plan → concurrent code → review → fix → verify)
src/agents.ts           role config + request-body builders
src/zerocache.ts        BYOK client; captures X-Zerocache-* headers; SSE assembly
src/rag.ts              in-memory vector index (cosine KNN)
src/trace.ts            run-trace event log, pricing, summarize/compare
src/render.ts / board.ts  the terminal board
src/diffapply.ts        unified-diff applier
src/verify.ts           runs `node --test` in the work tree
target-repo/            the pristine sample repo the agents edit (copied to .work/ per run)
traces/                 committed real-run traces
SCRIPT.md               the video narration script
```

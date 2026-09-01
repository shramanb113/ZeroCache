# Zerocache agent showcase

A bespoke multi-agent coding team ships a real change to a real repo, then
re-runs the **identical** task served entirely from Zerocache's cache:
byte-identical result, ~$0, a few seconds instead of a few minutes. A third
run with a **reworded** task still hits, through the semantic near-match tier.

This is the demo that answers "is this a fad?" — the agents do genuine
cross-file work and `node --test` goes red → green on screen; the cache just
makes the second pass free.

## What each run proves

| Run | Command | What you see |
| --- | --- | --- |
| **1 — cold** | `npm run -- run --run=1` | Real API calls. The team plans, three coders work in parallel, Claude reviews (streaming), a fixer applies feedback, tests pass. The ledger shows real upstream calls, tokens, and dollars. |
| **2 — warm** | `npm run -- run --run=2` | The same task. Every call returns `X-Zerocache-*-Hit: true`, the review stream replays instantly, the diffs are byte-identical, tests pass. Wall time drops to seconds, cost to `$0.00`. |
| **3 — semantic** | `npm run -- run --run=3` | The task, **reworded**. The architect/fixer calls still hit — `X-Zerocache-Completion-Hit-Kind: semantic` — because the final user message is within cosine threshold of run 1's. Needs a `--features semantic` Zerocache. |

`npm run -- run --run=2 --check` asserts the warm run had **zero** cache
misses and produced a byte-identical working tree; it exits non-zero
otherwise. `npm run -- run --record` runs 1 → 2 → 3 and prints the full
savings report.

## Zerocache surfaces exercised

| Stage | Agent | Route | Surface |
| --- | --- | --- | --- |
| Retrieve | Architect | `POST /openai/v1/embeddings`, `POST /gemini/v1/images/embeddings` | embeddings cache, **image**-embeddings cache |
| Plan | Architect | `POST /openai/v1/chat/completions` | chat completions cache, multi-provider |
| Repo brief | 3× Coder (parallel) | one **identical** `POST /openai/v1/chat/completions` fired concurrently | **in-process coalescing** (3 misses → 1 upstream call) |
| Implement | 3× Coder (parallel) | `POST /openai/v1/chat/completions` | chat completions cache |
| Review | Reviewer | `POST /anthropic/v1/messages` (streaming) | **native Anthropic messages cache**, **streaming replay** |
| Fix | Fixer | `POST /openai/v1/chat/completions` | chat completions cache |

Every request is BYOK — `Authorization: Bearer <that provider's key>`, never
a Zerocache key. `temperature: 0` throughout (also what the cache gate
requires).

## Prerequisites

- **`OPENAI_API_KEY`** — required (coders, architect, fixer, text embeddings).
- `ANTHROPIC_API_KEY` — optional. Without it the reviewer falls back to an
  OpenAI-wire model and the header shows `degraded: anthropic→openai`.
- `GEMINI_API_KEY` — optional. Without it the image-embedding step is skipped.
- A running Zerocache:
  ```sh
  cargo run -p zerocache-http                      # runs 1 and 2
  ZEROCACHE_SEMANTIC=1 cargo run -p zerocache-http --features semantic   # also run 3
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
savings report. The committed trace files are from a real pass — the README
and website quote them directly, nothing is hand-typed.

## Keyless smoke

`npm run smoke` starts `scripts/stub-zerocache.mjs` (a fake Zerocache that
needs no keys and no real instance) and runs the full pipeline against it —
useful for checking the rendering and the orchestration end-to-end in CI.
The stub's canned responses are not representative of real model output;
it only proves the plumbing.

## Layout

```
run.ts / warm.ts        CLI entry + cache pre-warm
src/orchestrator.ts     the agent team (plan → parallel code → review → fix → verify)
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

# completion-cache battle test

Proves the semantic LLM completion cache (CLAUDE.md Deviations item 21)
actually saves money for agent workloads, and prints the number.

## Run

```sh
# terminal 1: a real Zerocache instance
cargo run -p zerocache-http

# terminal 2
OPENAI_API_KEY=sk-... node demo/completion-cache/battle-test.mjs
```

No `npm install` — the script uses only global `fetch`. Override
`ZEROCACHE_BASE_URL` (default `http://localhost:8080`) or `MODEL` (default
`gpt-4o-mini`) as needed.

## What it proves

It measures by diffing `GET /metrics` (the
`zerocache_completion_*` counters) before and after each run — the same
technique `demo/mastra/battle-test.ts` uses for embeddings.

- **Phase 1 — a repeated agent run.** A scripted 4-turn triage loop (each
  turn built from the previous turn's real response, so the dependent
  chain only stays identical on a repeat because the cached replays are
  byte-exact). First run: 4 misses, real billed calls. Second run: 4
  hits, `X-Zerocache-Completion-Hit: true` on every turn, **zero** new
  provider calls — 100% off input and output for the whole loop.
- **Phase 2 — bulk auxiliary calls.** One short classify prompt sent 8
  times (the shape agents fire constantly): 1 miss + 7 hits.
- **Money.** Sums `prompt_tokens_saved` + `completion_tokens_saved` for
  the run and converts to USD at illustrative list price, then
  extrapolates to a 10k-runs/day fleet.

Each invocation folds a fresh `runId` into the system prompt, so it
starts cold for its own keys every time (the completion cache has no
`DELETE` route in v1) — re-running re-bills the cold phase.

## What it does not cover

Exact-match only — no semantic-similarity tier, no streaming, no real
tool execution. Those are deferred (see Deviations item 21).

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

---

# agent.mjs — a real tool-calling agent

`battle-test.mjs` uses a scripted multi-turn chain. `agent.mjs` is an
actual ReAct-style agent loop: the model picks a tool, the harness runs
it, feeds the result back, and repeats until the model produces a final
answer. Every step is a real `/v1/chat/completions` call with a `tools`
array at `temperature: 0`.

Scenario: a support-triage agent with three tools over in-script data
(`search_kb`, `get_order`, `resolve_ticket`) resolving a 3-ticket suite.

```sh
# terminal 1 — point Zerocache at any OpenAI-wire-compatible endpoint
ZEROCACHE_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
  cargo run -p zerocache-http

# terminal 2
OPENAI_API_KEY=<key> MODEL=gemini-3.5-flash-lite PACE_MS=5000 \
  node demo/completion-cache/agent.mjs
```

`MODEL` defaults to `gpt-4o-mini` (expects a real OpenAI key + default
base URL). `PACE_MS` (default 7000) spaces out **cold-run** calls so a
provider free-tier per-minute quota doesn't trip; the repeat run is all
cache hits and never waits.

## The money case it demonstrates

A CI / eval loop: the same ticket suite runs on every commit. Run 1 pays
full price. Run 2 — identical prompts, tool definitions, and model — is
served **entirely** from the completion cache, including every
intermediate tool-call turn: zero upstream calls, 100% off input and
output tokens, byte-identical resolutions.

Measured against live Gemini (`gemini-3.5-flash-lite`), a 3-ticket run:
~9 model calls + 9 tool calls on the cold run; the repeat run reused all
9 and re-billed nothing (~4.4k prompt + ~355 completion tokens saved per
re-run). It scales with suite size, run frequency, and model price.

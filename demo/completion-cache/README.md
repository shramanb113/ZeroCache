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
`gpt-4o-mini`) as needed. Any of the built-in chat providers (`openai`,
`mistral`, `gemini`, `groq`, `deepseek`, `together`, `openrouter`, `xai`,
`fireworks`) works with no server config; set `ZEROCACHE_CHAT_PROVIDER` to
pick which one the script calls. To run against Gemini's OpenAI-compat
endpoint:

```sh
# terminal 1
ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" \
  cargo run -p zerocache-http

# terminal 2
OPENAI_API_KEY=<gemini key> MODEL=gemini-3.5-flash-lite ZEROCACHE_CHAT_PROVIDER=gemini \
  node demo/completion-cache/battle-test.mjs
```

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

`battle-test.mjs` / `agent.mjs` are exact-match only — no streaming, no real
tool execution (deferred, Deviations item 21). The semantic tier has its own
script below.

---

# paraphrase.mjs — the opt-in semantic tier

Proves Deviations item 24: the same deterministic question, reworded five
ways, is served from cache on repeat; a materially different question is not.
Needs a Zerocache built with `--features semantic` and run with
`ZEROCACHE_SEMANTIC=1` on the sled backend, plus a real key.

```sh
ZEROCACHE_SEMANTIC=1 \
  ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" \
  cargo run -p zerocache-http --features semantic

OPENAI_API_KEY=<gemini key> MODEL=gemini-3.5-flash-lite ZEROCACHE_CHAT_PROVIDER=gemini \
  node demo/completion-cache/paraphrase.mjs
```

Asserts: base question is a cold miss, ≥ 4/5 paraphrases return
`X-Zerocache-Completion-Hit-Kind: semantic` with a byte-identical body, the
different question does not hit, and the `zerocache_completion_semantic_hits_total`
delta is ≥ 4 while misses rose by exactly 2. If fewer than 4/5 hit at the
default `0.97`, lower `ZEROCACHE_SEMANTIC_THRESHOLD` for the run (not the code
default) or tighten the paraphrases.

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
# terminal 1 — register Gemini as a chat provider (no doubled /v1)
ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" \
  cargo run -p zerocache-http

# terminal 2
OPENAI_API_KEY=<key> MODEL=gemini-3.5-flash-lite PACE_MS=5000 ZEROCACHE_CHAT_PROVIDER=gemini \
  node demo/completion-cache/agent.mjs
```

`MODEL` defaults to `gpt-4o-mini` and `ZEROCACHE_CHAT_PROVIDER` to `openai`
(expects a real OpenAI key). `PACE_MS` (default 7000) spaces out
**cold-run** calls so a provider free-tier per-minute quota doesn't trip;
the repeat run is all cache hits and never waits.

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

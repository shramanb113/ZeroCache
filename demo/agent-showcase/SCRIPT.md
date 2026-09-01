# Zerocache — agent showcase video script

> **Audience:** founders and engineering leaders at AI-product startups (SF / NY /
> London / Europe), and the people who hire for those teams. They have seen a
> hundred "look at my agent" demos. This one has to (a) prove the agent does
> real work, (b) show a problem they personally feel, and (c) show it solved in
> a way nothing else on the market does.
>
> **Runtime:** ~6:00 main cut. A 90-second cut is at the bottom, plus two
> optional appendices (cross-replica coalescing; the tamper test).
>
> **Numbers:** every `⟦TOKEN⟧` below is filled from the committed trace files
> (`traces/run-{cold,warm,semantic}.jsonl`) via `src/trace.ts` `compare()`.
> Do not hand-type them. The table at the very bottom is the single source of
> truth; fill it once, then find/replace the tokens in this script.
>
> **Format:** each beat is `VOICEOVER` (what you say) + `ON SCREEN` (what the
> viewer sees). Keep the terminal at ~110 cols, large font, dark theme. Colour
> is on (`COLOR=1`, a real TTY).

---

## Reference run (what to expect on screen)

Measured on the reference pass — `gpt-4o-mini` coders/architect/fixer,
`claude-sonnet-5` reviewer on native `/v1/messages`, `text-embedding-3-small`
retrieval, OpenAI + Anthropic keys, image step off:

| | Run 1 (cold) | Run 2 (warm) | Run 3 (reworded) |
| --- | --- | --- | --- |
| upstream calls | `⟦COLD_CALLS⟧` | **0** | `⟦SEM_CALLS⟧` |
| prompt tokens billed | `⟦COLD_PROMPT_TOK⟧` | **0** | `⟦SEM_PROMPT_TOK⟧` |
| completion tokens billed | `⟦COLD_COMPL_TOK⟧` | **0** | `⟦SEM_COMPL_TOK⟧` |
| wall time | `⟦COLD_WALL⟧` | `⟦WARM_WALL⟧` | `⟦SEM_WALL⟧` |
| est. cost | `⟦COLD_USD⟧` | **$0.00** | `⟦SEM_USD⟧` |
| cache hits | 0 | `⟦WARM_HITS⟧` | `⟦SEM_HITS⟧` |
| semantic hits | 0 | 0 | `⟦SEM_HITS⟧` |
| coalesced calls | `⟦COALESCED⟧` | — | — |
| `node --test` | `⟦COLD_TESTS⟧` passed | same, byte-identical | same |

> An earlier pass run entirely through **Gemini** (`gemini-2.5-flash`) showed the
> same *shape*: cold ≈ 13 upstream calls / ≈ 23k tokens / ≈ 24 s, warm = 0 / 0 /
> ~0.2 s, byte-identical, `CHECK PASSED`. Use the OpenAI+Anthropic numbers above
> for the video so the dollar figure is real.

---

## MAIN CUT

### 0:00 — Cold open (the hook)

**VOICEOVER:**
"This is an autonomous engineering team. It's going to take a real feature
request, write the code across four files, have Claude review it, fix what
Claude flags, and run the tests — for real, with real API calls. Watch."

**ON SCREEN:**
- Black screen, one line of text fades in: **"Run it once. Every run after that is free."**
- Cut straight to the terminal. `npm run -- run --run=1` already typed. Hit enter.
- The board renders: header (`ZEROCACHE · agent showcase`, `run 1 of 3 · cold ·
  providers: openai · anthropic`), seven stage rows all `·` pending.

---

### 0:18 — The problem, made big

**VOICEOVER:**
"First, why this matters. If you're shipping anything with LLMs, you are paying
for the same work over and over. Your eval suite runs in CI a few hundred times
a week — same prompts, same deterministic outputs, full price every time. You
tweak one line of a prompt and re-run the whole agent. Your multi-agent system
fans out near-identical prompts across a dozen workers. A provider hiccups and
your retry logic re-bills the entire call. And every engineer on the team is
re-running the same flows locally all day.

None of that is a rounding error. For a team doing serious agent work it's a
real line on the bill and a real tax on how fast you can iterate. The second
time you run a deterministic LLM workload, you should not be paying for it
again. Today, you are."

**ON SCREEN:**
- Left third: the live terminal keeps running (RETRIEVE / PLAN light up).
- Right two-thirds: a plain list builds one line at a time as you name each one —
  `CI eval loops`, `prompt iteration`, `multi-agent fan-out`, `retry storms`,
  `local dev re-runs`. Under it: **"pay full price every time."**

---

### 1:05 — What exists today, and why it doesn't close the gap

**VOICEOVER:**
"You'd think this is solved. It isn't.

Provider-side prompt caching — OpenAI's automatic discount, Anthropic's
cache_control — only discounts the *input* tokens. The request still runs, you
still pay for every output token, and it evicts in five to sixty minutes. It
does nothing for a run you do tomorrow.

Framework caches — LangChain's, LlamaIndex's — live inside one Python process,
match only byte-identical strings, and die when the process dies.

And the hosted semantic-cache products mean your prompts and your completions
leave your infrastructure and land in someone else's database. For most of the
enterprise deals you're chasing, that's an instant no.

What nobody ships: a provider-neutral, self-hosted cache you own, that does
both exact and semantic matching, that you put in front of any model with a
one-line change and no SDK."

**ON SCREEN:**
- Three-row table, each row appears as you say it:
  | | catch |
  |---|---|
  | provider prompt caching | input tokens only · still runs · evicts in minutes |
  | framework caches | one process · exact match · in-memory |
  | semantic-cache SaaS | your prompts leave your infra |
- Then a fourth row drops in, highlighted: **`Zerocache` — self-hosted · provider-neutral · exact + semantic · BYOK · one static binary**

---

### 1:45 — What Zerocache is

**VOICEOVER:**
"Zerocache is a caching proxy written in Rust. It ships as one 15-megabyte
static binary — no runtime, no libc, nothing. You point your existing OpenAI or
Anthropic client at it by changing the base URL. That's the entire integration.

Every request carries your own provider key — Zerocache never holds a
credential, it just hashes the key so one caller's cache is never another
caller's. It caches chat completions, Anthropic messages, and embeddings; it
handles streaming; and with one flag it also matches requests that mean the
same thing but aren't worded identically. All of that is what this demo
exercises."

**ON SCREEN:**
- One diagram: `your agent ──base_url──▶ Zerocache ──BYOK──▶ OpenAI / Anthropic / Gemini`
- with a side box: `sled or Redis · Prometheus /metrics · live /dashboard`
- Terminal (left) is now into IMPLEMENT — coder rows ticking over. Let it breathe.

---

### 2:10 — Run 1, cold: the team ships real code

**VOICEOVER:**
"Here's run one, live. Nothing is cached yet.

RETRIEVE — it chunks the sample repo and embeds it through Zerocache so the
architect has context. PLAN — the architect writes an implementation plan.

BRIEF — and this is the first Zerocache feature. Three coder workers all ask
for the same repo summary at the same moment. That's one identical request
fired three times concurrently. Zerocache collapses them into a *single*
upstream call and fans the answer back to all three. Three-to-one, before the
cache even has an entry.

IMPLEMENT — the coders write `rateLimit.ts`, wire it into the routes, add the
config knob, write the tests. Four files.

REVIEW — this goes to Claude, on its *native* messages API, not an OpenAI-shaped
shim — and it's streamed. You're watching the review token-by-token as it
arrives.

FIX — the fixer takes Claude's feedback and rewrites the files it flagged.

VERIFY — `node --test`, in the work tree, for real."

**ON SCREEN:**
- Each stage flips `·` → `●` → `✓` as narrated. On BRIEF, the detail reads
  **`3 workers → 1 upstream call (coalesced)`** — pause on it.
- On REVIEW, the streaming review text scrolls in the lower panel. Let a couple
  of real sentences land.
- On VERIFY: **`⟦COLD_TESTS⟧ passed, 0 failed`** in green.
- LEDGER panel settles: `upstream calls ⟦COLD_CALLS⟧ · prompt tokens
  ⟦COLD_PROMPT_TOK⟧ · completion ⟦COLD_COMPL_TOK⟧ · wall ⟦COLD_WALL⟧ · est. cost
  ⟦COLD_USD⟧`.

**VOICEOVER (over the ledger):**
"Real work: `⟦COLD_CALLS⟧` upstream calls, `⟦COLD_USD⟧`, `⟦COLD_WALL⟧`. The
agent did the job. Now the point of the whole demo."

---

### 3:35 — Run 2, warm: the second run is free

**VOICEOVER:**
"Same task. Same command. Nothing about the request changed. Watch the stage
markers."

**ON SCREEN:**
- Type `npm run -- run --run=2`. Enter.
- Every stage resolves to **`⚡`** almost instantly. Details read `exact · 0 ms
  billed`, `replayed from cache`. BRIEF reads `3 workers · all cached`.
- The REVIEW panel replays the *same* Claude review as a stream — visibly fast.
- VERIFY: **`⟦COLD_TESTS⟧ passed, 0 failed`** — same numbers.
- Then the CHECK line: **`CHECK PASSED: warm run was all hits and byte-identical.`**
- SAVINGS REPORT table renders: `upstream calls ⟦COLD_CALLS⟧ → 0`, `est. cost
  ⟦COLD_USD⟧ → $0.00`, `wall time ⟦COLD_WALL⟧ → ⟦WARM_WALL⟧`.

**VOICEOVER:**
"Zero upstream calls. Zero dollars. `⟦WARM_WALL⟧` instead of `⟦COLD_WALL⟧`. And
the working tree it produced is byte-for-byte identical to run one — the
`--check` flag asserts that and fails the run if it's off by a character.
Including the streamed Claude review, which replayed from cache as a stream.

This is what 'the second run is free' actually means. Your CI eval loop, your
prompt-tuning iteration, your re-run after a flaky test — the repeat is
instant and costs nothing, and you can prove it's the same output."

---

### 4:20 — Run 3: reworded request, semantic near-match

**VOICEOVER:**
"But real life isn't byte-identical. Someone rewrites the ticket. A teammate
asks the same thing in their own words. Exact-match caching gives up here.
Zerocache doesn't have to."

**ON SCREEN:**
- Show the two briefs side by side, highlight that they share no sentence:
  - run 1: *"Add per-API-key token-bucket rate limiting to the links API: 60
    requests per minute per `X-Api-Key`…"*
  - run 3: *"Clients are hammering the links API. Put a cap on how many requests
    each API key can make per minute (sixty)…"*
- Type `npm run -- run --run=3`. Enter.
- Stages resolve to `⚡` with detail **`semantic · score ⟦SEM_SCORE⟧`** and the
  header line `X-Zerocache-Completion-Hit-Kind: semantic`.
- SAVINGS REPORT: `semantic hits ⟦SEM_HITS⟧`, `est. cost ⟦SEM_USD⟧`.

**VOICEOVER:**
"Different words, same intent — `⟦SEM_HITS⟧` of those calls served from run
one's cache. Here's the part that matters for trust: the similarity is computed
by a small embedding model running *inside* Zerocache. No extra API call, no
second key, and not one byte of your prompt leaves the box. It's gated by a
deliberately conservative threshold, so it errs toward a miss — toward doing
the real call — rather than toward a confident wrong answer. And a matched
completion is still a real, previously-served completion, never a paraphrase."

---

### 5:05 — The dashboard

**VOICEOVER:**
"None of this is a black box. Zerocache serves a live dashboard — hit rate,
tokens you weren't billed for, dollars avoided, per-provider — straight off the
same metrics endpoint Prometheus scrapes. This is the number you put in front
of whoever signs off on the infra bill."

**ON SCREEN:**
- Browser: `localhost:8080/dashboard`. The cost-avoided line and per-provider
  table, updating live. Nudge a price field and show it recompute instantly.

---

### 5:25 — Why it's safe to trust

**VOICEOVER:**
"Two things people always ask. One: what happens when I upgrade the model? The
model's identity is part of the cache key. A new model, or a new version of the
adapter, can't collide with the old entries — they just become unreachable. You
can't be served a stale answer from the model you stopped using.

Two: does it ever cache a failure? No. Only a successful, deterministic
response is stored. An error is passed straight through with its real status
code and forgotten."

**ON SCREEN:**
- Two short lines, typewritten:
  - `key = blake3(owner · provider · scope · model · adapter_version · request)`
  - `only 2xx · only deterministic · absent, never wrong`

---

### 5:45 — Close

**VOICEOVER:**
"An autonomous team that ships real, tested code — and a cache that makes every
run after the first one free, provable, and yours. It's Rust, it's one binary,
it's open. Links below. If you're building here, let's talk."

**ON SCREEN:**
- Split: left, the SAVINGS REPORT table frozen; right,
  `github.com/shramanb113/ZeroCache` + your name + contact.
- Fade to the opening line: **"Run it once. Every run after that is free."**

---

## 90-SECOND CUT

**0:00** — "This is an autonomous engineering team. Real feature request, four
files, Claude reviews it, tests run — all real API calls." *(board running)*

**0:12** — "Every AI team pays for the same work twice: CI eval loops, prompt
iteration, multi-agent fan-out, retries. Provider prompt caching only discounts
input tokens and expires in minutes. Framework caches die with the process.
Hosted ones take your prompts off your infra." *(4-row table)*

**0:30** — "Zerocache: a Rust proxy, one static binary, one base-URL change.
Run one, cold — `⟦COLD_CALLS⟧` calls, `⟦COLD_USD⟧`, `⟦COLD_WALL⟧`, tests green.
Note BRIEF: three concurrent identical calls collapsed to one." *(ledger)*

**0:52** — "Run two, same task: every stage a cache hit, byte-identical output,
`$0.00`, `⟦WARM_WALL⟧`. The `--check` flag fails the run if it's off by a
character." *(all ⚡ + CHECK PASSED)*

**1:10** — "Run three, the ticket *reworded* — no shared sentence. `⟦SEM_HITS⟧`
calls still hit, matched by a local embedding model, nothing leaving the box,
conservative threshold." *(semantic header)*

**1:25** — "Real agent, real tests, and every run after the first is free and
provable. Rust, open, link below."

---

## SHOT LIST / CAPTURE CHECKLIST

1. **Terminal**, 110×32, big font, dark, `COLOR=1`. Record runs 1→2→3 in one
   unbroken session so the cache state is honest.
2. Before recording run 2, do **not** clear the store. Before run 3, restart on
   `--features semantic` with `ZEROCACHE_SEMANTIC=1` (mention this cut or hide it).
3. **Grab stills** of: the BRIEF `3 → 1 coalesced` line; the streaming REVIEW
   panel mid-stream; the `CHECK PASSED` line; the SAVINGS REPORT.
4. **Browser**: `/dashboard` with real traffic already through it; one live
   price edit.
5. **Side-by-side briefs** (run 1 vs run 3) as a static graphic — pull the exact
   strings from `src/agents.ts` (`RATE_LIMIT_TASK.brief` / `RATE_LIMIT_TASK_PARAPHRASED.brief`).
6. Optional B-roll: `ls -la target/release/zerocache-http` showing the ~15 MB
   size; `docker images` showing the `FROM scratch` image.

---

## OPTIONAL APPENDIX A — cross-replica coalescing (≈45 s)

**VOICEOVER:**
"One more, for anyone running this at scale. In-process coalescing is per
process. Put Zerocache behind a load balancer with three replicas and the same
brand-new request could still hit your provider once per replica. Turn on
`ZEROCACHE_CROSS_REPLICA_COALESCING` on the Redis backend and a request that
resolves to a single key is single-flighted *across* replicas through a Redis
lock — three pods, one upstream call, the other two wait and read the fill."

**ON SCREEN:**
- Three terminal panes, same `curl` to a shared LB at the same instant.
- `zerocache_cross_replica_coalesced_total{kind="completion"}` ticks to 2 on
  `/metrics`; only one pod logs an upstream call.

---

## OPTIONAL APPENDIX B — the tamper test (≈30 s)

**VOICEOVER:**
"It's not record-and-replay. Change one line of the task and re-run: only the
calls downstream of that change re-bill. Everything byte-identical to before
still comes from cache. Content-addressed, so partial by construction."

**ON SCREEN:**
- Edit one line of `RATE_LIMIT_TASK.brief`, `--run=1` against a warm store.
- Ledger shows a handful of misses, not a full run; SAVINGS still shows most
  calls hitting.

---

## NUMBERS TO FILL (from `traces/`, via `npm run -- run --record`)

Run `compare(run-cold, run-warm, run-semantic)` (it prints as the SAVINGS
REPORT) and copy:

| token | source | value |
| --- | --- | --- |
| `⟦COLD_CALLS⟧` | run 1 · upstream calls | |
| `⟦COLD_PROMPT_TOK⟧` | run 1 · prompt tokens billed | |
| `⟦COLD_COMPL_TOK⟧` | run 1 · completion tokens billed | |
| `⟦COLD_WALL⟧` | run 1 · wall time | |
| `⟦COLD_USD⟧` | run 1 · est. cost | |
| `⟦COLD_TESTS⟧` | run 1 · `node --test` pass count | |
| `⟦COALESCED⟧` | run 1 · coalesced calls | |
| `⟦WARM_WALL⟧` | run 2 · wall time | |
| `⟦WARM_HITS⟧` | run 2 · cache hits | |
| `⟦SEM_CALLS⟧` | run 3 · upstream calls | |
| `⟦SEM_PROMPT_TOK⟧` | run 3 · prompt tokens billed | |
| `⟦SEM_COMPL_TOK⟧` | run 3 · completion tokens billed | |
| `⟦SEM_WALL⟧` | run 3 · wall time | |
| `⟦SEM_USD⟧` | run 3 · est. cost | |
| `⟦SEM_HITS⟧` | run 3 · semantic hits | |
| `⟦SEM_SCORE⟧` | any run-3 semantic hit · `X-Zerocache-Semantic-Score` | |

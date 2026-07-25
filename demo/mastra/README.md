# Zerocache + Mastra: agentic multimodal RAG demo

This demo proves that Zerocache works as a drop-in embedding cache inside a real,
production-shaped **agentic** RAG pipeline — not just a raw embedding client calling
`/v1/embeddings` directly (that was already covered by `demo/langchain-ts` and
`demo/llamaindex-python`). Here, an LLM ([Mastra](https://mastra.ai) `Agent`) decides
*whether and which* tool to call, and each tool independently embeds through Zerocache:

- **Text** documents are embedded via **OpenAI** (`text-embedding-3-small`), through
  Zerocache's `/openai/v1/embeddings`.
- **Images** are embedded via **Gemini** (`gemini-embedding-2`), through Zerocache's
  `/gemini/v1/images/embeddings` — the multimodal image-embedding capability added to
  Zerocache in Tasks 1–8 of this plan (see `CLAUDE.md`'s Deviations items 14–15).

The two embedding spaces are incomparable (different models, different dimensionality),
so they're kept in two separate vector indexes (`src/mastra/rag/vector-store.ts`), and
the agent has two separate tools — `searchDocuments` and `searchImages`
(`src/mastra/tools/`) — rather than one combined tool, specifically so tool selection is
a real model decision, not a foregone conclusion.

**Image-embedding support is Gemini-only.** `openai`, `mistral`, and `huggingface` don't
implement `ImageEmbeddingProvider` and correctly `404` on the images endpoint — see
`CLAUDE.md`'s Deviations item 14 ("Multimodal image embeddings ... a separate
`ImageEmbeddingProvider` port, implemented by Gemini only") for the full reasoning
(OpenAI's public API has no image-embedding endpoint at all as of the live check at
planning time).

## Prerequisites

- `OPENAI_API_KEY` — a real OpenAI API key (used for text embeddings and for the agent's
  chat model, `gpt-4o-mini`).
- `GEMINI_API_KEY` — a real Gemini API key (used for image embeddings).
- A running Zerocache instance. From the repo root:

  ```sh
  cargo run -p zerocache-http
  ```

  By default the demo talks to `http://localhost:8080` (override with
  `ZEROCACHE_BASE_URL`).
- Node.js `>= 22.13.0` (see `engines` in `package.json`) and dependencies installed:

  ```sh
  npm install
  ```

## Running it

All commands below are run from `demo/mastra/`.

**Ingest the sample knowledge base** (populates the LibSQL-backed vector store from
`sample-data/v1/` — 6 Aurora Cloud Storage docs + 2 generated PNGs):

```ts
import { ingestSampleData } from "./src/mastra/rag/ingest";
await ingestSampleData("sample-data/v1");
```

(There isn't a dedicated ingest CLI script — `battle-test.ts`, below, exercises this
directly, including re-ingesting `sample-data/v2/` to show a realistic edit+addition.)

**Run the agent** — `mastra dev` starts Mastra's local dev server/playground, from
which `ragAgent` can be exercised interactively (ask it a pricing question, or attach
an image and ask what it shows):

```sh
npm run dev
```

**Run the unit tests** (no API keys required — these exercise the plain-fetch Zerocache
client and other logic against mocks):

```sh
npx vitest run
```

**Run the end-to-end battle test** (requires real `OPENAI_API_KEY`/`GEMINI_API_KEY` and
a running Zerocache instance — makes real, billed provider calls):

```sh
npx tsx battle-test.ts
```

`battle-test.ts` primes a cold cache itself (deletes every entry Parts A-C could have
created, from both `v1/` and `v2/`) before Part A runs, so its exact hit/miss assertions
(Check 1: 8 misses; Check 2: 8 hits) hold whether this is the first run ever against your
Zerocache instance or the fifth — you don't need a freshly wiped store to reproduce the
18/18 result below. Re-running does make real, billed embed/generate calls again (the
priming deletes themselves are free — no provider call), so there's a real cost to running
it repeatedly, just not a risk of spurious failures from stale cache state.

## Why the cache actually helps here

The sample knowledge base ships in two generations to make the cache benefit concrete
and measurable rather than illustrative:

- `sample-data/v1/` — 6 text docs + 2 images (8 items total).
- `sample-data/v2/` — the same 8 items, with `pricing.md` edited (updated pricing:
  $12/month, 750GB, up from v1's $9/month, 500GB) and one new file,
  `bulk-export-feature.md`, added (9 total). Every other file is byte-identical to v1.

`battle-test.ts`'s Part A ingests v1 cold, re-ingests v1 unchanged, then ingests v2, and
diffs Zerocache's `/metrics` hit/miss counters around each run. Here is the real summary
table from an actual run against a live Zerocache instance with real OpenAI/Gemini keys
(2026-07-25, 18/18 checks passed):

```
Run                       Items  Hits  Misses  Tokens billed  Duration
1. Cold ingest (v1)       8      0     8       516            8535ms
2. Rebuild, unchanged     8      8     0       0              1128ms
3. Rebuild, 1 edit+1 new  9      7     2       189            2751ms
```

Run 2 shows the ideal case — a full unchanged re-index costs 0 tokens and drops from
8.5s to ~1.1s. Run 3 is the realistic case: re-indexing after one edited doc and one new
doc costs exactly 2 misses (the edited `pricing.md` and the new
`bulk-export-feature.md`) out of 9 items — the other 7, byte-identical to v1, hit the
cache for free.

All four agentic-behavior checks also passed on that same run: correct tool selection
for a text question (`searchDocuments` called, `searchImages` not, and the answer
reflected v2's *edited* pricing — not v1's stale numbers); correct tool selection for an
image question (`searchImages` called, `searchDocuments` not, correctly distinguishing
`architecture-diagram.png` from `dashboard-screenshot.png`); multi-hop synthesis (a
question spanning both `pricing.md` and `bulk-export-feature.md` answered completely);
and judgment not to retrieve (an off-topic question triggered zero tool calls and no
hallucinated content). Delete/re-embed roundtrips for both text and image also passed —
the first HTTP-level exercise of Zerocache's `DELETE` routes, previously only
Rust-unit-tested.

**What this demo does *not* claim:** per `CLAUDE.md`'s non-goals ("Live/conversational
query embedding caching (deferred — low reuse rate, unproven)"), the benefit measured
here is **ingestion-pipeline reuse**, not repeated-query caching. The agent's own query
embeddings (`searchDocuments`/`searchImages` embedding the user's question each time)
are expected to mostly miss — every real question is a new string — and that's fine,
it's not what this demo is measuring. The thing being validated is that re-indexing an
evolving corpus doesn't re-pay for content that hasn't changed.

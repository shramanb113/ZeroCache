// End-to-end battle test for the Mastra RAG demo -- the culminating exercise of this
// plan, driving Tasks 9-11's real exports (not mocks) against a real running Zerocache
// instance and real OPENAI_API_KEY/GEMINI_API_KEY. Following this session's established
// battle-test pattern (see demo/langchain-ts/src/battle-test.ts and
// demo/llamaindex-python/src/battle_test.py for the style this mirrors).
//
// Three parts:
//   Part A -- the cache-benefit story: cold ingest (all misses), an identical rebuild
//             (all hits, free), and a realistic 1-edit-1-new-file rebuild (exactly 2
//             misses, 7 hits). ingestSampleData() (Task 10) does not itself return
//             hit/miss counts, so this script derives them the only way an outside
//             caller can: diffing zerocache_cache_hits_total/zerocache_cache_misses_total
//             on /metrics before and after each run.
//   Part B -- proves this is agentic RAG, not a fixed retrieve-then-answer pipeline, by
//             asserting on the agent's actual tool-call trace (Agent.generate()'s
//             toolCalls/toolResults arrays -- see node_modules/@mastra/core's own
//             embedded docs, reference-agents-generate.md, "Response structure": tool
//             data is wrapped in `payload`, e.g. `toolCall.payload.toolName`), not just
//             the final answer text.
//   Part C -- delete, HTTP-level: the only end-to-end exercise of either delete route
//             (Task 5's delete_batch/delete_image_batch otherwise only have Rust unit
//             tests).
//
// Requires a real running Zerocache instance (ZEROCACHE_BASE_URL, default
// http://localhost:8080) and real OPENAI_API_KEY/GEMINI_API_KEY -- this makes real,
// billed provider calls. Run with: npx tsx battle-test.ts
//
// Safe to re-run: main() primes a cold cache first (see primeColdCache below) by
// deleting every entry Parts A-C's ingestion could have created, so Check 1/2's exact hit/miss
// counts hold on every invocation against a persistent store, not just the first
// one ever made. Priming uses only DELETE calls (pure cache-key computation, no
// provider call, no billing), so re-running does re-bill the real embed/generate
// calls Parts A-C make, but not an extra ingestion's worth on top of that.
//
// Note: importing ingestSampleData (Task 10) pulls in src/mastra/rag/ingest.ts, which
// itself calls requireEnv("OPENAI_API_KEY")/requireEnv("GEMINI_API_KEY") at module load
// time (deliberately, per that file's own comments) -- so a missing key surfaces as an
// import-time crash before main() even runs, not a graceful SKIP like the langchain-ts/
// llamaindex-python scripts' key-independent sections. That's consistent with this
// task's premise: Part A-C all require real keys, there is no key-independent subset.

import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

import { ingestSampleData } from "./src/mastra/rag/ingest";
import {
  embedText,
  embedImage,
  deleteText,
  deleteImage,
} from "./src/mastra/zerocache/client";
import { ZEROCACHE_BASE_URL, requireEnv } from "./src/mastra/zerocache/env";
import { ragAgent } from "./src/mastra/agents/rag-agent";
import { RequestContext } from "@mastra/core/request-context";
import type { SearchImagesRequestContext } from "./src/mastra/tools/search-images";

const OPENAI_API_KEY = requireEnv("OPENAI_API_KEY");
const GEMINI_API_KEY = requireEnv("GEMINI_API_KEY");

// Must match src/mastra/rag/ingest.ts and the searchDocuments/searchImages tools --
// Part C reconstructs the exact same (provider, model, input) so its delete calls
// target the exact cache entries Run 1/2/3 populated.
const TEXT_MODEL = "text-embedding-3-small";
const IMAGE_MODEL = "gemini-embedding-2";

const SAMPLE_V1 = "sample-data/v1";
const SAMPLE_V2 = "sample-data/v2";

type Result = "PASS" | "FAIL";
const results: { name: string; result: Result; detail?: string }[] = [];

function record(name: string, result: Result, detail?: string) {
  results.push({ name, result, detail });
  const marker = { PASS: "✓", FAIL: "✗" }[result];
  console.log(`${marker} [${result}] ${name}${detail ? " -- " + detail : ""}`);
}

async function run(name: string, fn: () => Promise<void>) {
  try {
    await fn();
  } catch (err) {
    record(name, "FAIL", err instanceof Error ? err.message : String(err));
  }
}

// ---------- /metrics helpers (Part A) ----------

async function fetchMetricsText(): Promise<string> {
  const res = await fetch(`${ZEROCACHE_BASE_URL}/metrics`);
  return res.text();
}

// Sums every line for a given metric family whose label set contains all of `filters`
// (substring match on `key="value"` inside the {...} block) -- robust to label order
// (zerocache-http/src/app.rs emits content_type before provider) and to label
// combinations that don't exist yet on a fresh /metrics scrape (sums to 0, the correct
// "nothing recorded yet" default).
function sumMetric(
  metricsText: string,
  family: string,
  filters: Record<string, string> = {},
): number {
  let sum = 0;
  for (const line of metricsText.split("\n")) {
    if (!line.startsWith(`${family}{`)) continue;
    const match = line.match(/\{([^}]*)\}\s+([0-9.eE+-]+)\s*$/);
    if (!match) continue;
    const [, labels, valueStr] = match;
    const matchesAllFilters = Object.entries(filters).every(([k, v]) =>
      labels.includes(`${k}="${v}"`),
    );
    if (!matchesAllFilters) continue;
    const value = Number(valueStr);
    if (!Number.isNaN(value)) sum += value;
  }
  return sum;
}

interface HitsMissesTokens {
  hits: number;
  misses: number;
  tokensOpenai: number;
  tokensGemini: number;
}

async function fetchHitsMissesTokens(): Promise<HitsMissesTokens> {
  const text = await fetchMetricsText();
  return {
    hits: sumMetric(text, "zerocache_cache_hits_total"),
    misses: sumMetric(text, "zerocache_cache_misses_total"),
    tokensOpenai: sumMetric(text, "zerocache_provider_prompt_tokens_total", {
      provider: "openai",
    }),
    tokensGemini: sumMetric(text, "zerocache_provider_prompt_tokens_total", {
      provider: "gemini",
    }),
  };
}

interface RunRow {
  tableLabel: string;
  items: number;
  hits: number;
  misses: number;
  tokens: number;
  durationMs: number;
}

async function runIngest(
  checkLabel: string,
  tableLabel: string,
  dir: string,
  expected: { hits: number; misses: number },
  opts?: { expectZeroTokens?: boolean },
): Promise<RunRow> {
  const before = await fetchHitsMissesTokens();
  const start = performance.now();
  const { textDocs, images } = await ingestSampleData(dir);
  const durationMs = performance.now() - start;
  const after = await fetchHitsMissesTokens();

  const hits = after.hits - before.hits;
  const misses = after.misses - before.misses;
  const tokens =
    after.tokensOpenai -
    before.tokensOpenai +
    (after.tokensGemini - before.tokensGemini);
  const items = textDocs + images;

  const ok = hits === expected.hits && misses === expected.misses;
  record(
    checkLabel,
    ok ? "PASS" : "FAIL",
    `items=${items} hits=${hits} misses=${misses} (expected hits=${expected.hits} misses=${expected.misses})`,
  );

  if (opts?.expectZeroTokens) {
    record(
      `${checkLabel} -- 0 additional tokens billed`,
      tokens === 0 ? "PASS" : "FAIL",
      `tokens billed this run=${tokens}`,
    );
  }

  return { tableLabel, items, hits, misses, tokens, durationMs };
}

function printSummaryTable(rows: RunRow[]): void {
  const header = [
    "Run",
    "Items",
    "Hits",
    "Misses",
    "Tokens billed",
    "Duration",
  ];
  const widths = [26, 7, 6, 8, 15, 10];
  const fmtRow = (cells: string[]) =>
    cells.map((c, i) => c.padEnd(widths[i])).join("");
  console.log("\n" + fmtRow(header));
  for (const row of rows) {
    console.log(
      fmtRow([
        row.tableLabel,
        String(row.items),
        String(row.hits),
        String(row.misses),
        String(row.tokens),
        `${Math.round(row.durationMs)}ms`,
      ]),
    );
  }
}

// ---------- agent tool-call trace helpers (Part B) ----------

interface SearchDocumentsResult {
  results: { id: string; score: number; file?: string; text?: string }[];
}

interface SearchImagesResult {
  results: { id: string; score: number; file?: string }[];
}

// Structurally typed against whatever Agent.generate() actually returns (FullOutput<T>
// from @mastra/core/dist/stream/base/output.d.ts) rather than importing that internal
// type by name -- avoids coupling this script to Mastra's internal type export paths,
// and Awaited<ReturnType<typeof ragAgent.generate>> is structurally assignable to this
// regardless of which overload TS's ReturnType<T> resolves against.
interface AgentTrace {
  text: string;
  toolCalls: { payload: { toolName: string } }[];
  toolResults: { payload: { toolName: string; result: unknown } }[];
}

function calledTools(trace: AgentTrace): string[] {
  return trace.toolCalls.map((tc) => tc.payload.toolName);
}

function toolResultPayloads<T>(trace: AgentTrace, toolName: string): T[] {
  return trace.toolResults
    .filter((tr) => tr.payload.toolName === toolName)
    .map((tr) => tr.payload.result as T);
}

// Builds the (prompt, requestContext) pair for an image query. The image bytes travel
// via requestContext, NOT as base64 text embedded in the prompt -- see
// src/mastra/tools/search-images.ts's comment for why: this battle test's first version
// asked the model to retype a base64 blob verbatim as a tool argument, and gpt-4o-mini
// reproducibly corrupted it (624 -> 600 chars, mismatch) on every run, a real bug this
// task's Step 2 live run caught and Task 11's searchImagesTool was fixed for.
async function buildImageQuery(
  sampleDataDir: string,
  file: string,
  mimeType: string,
): Promise<{ prompt: string; requestContext: RequestContext<SearchImagesRequestContext> }> {
  const bytes = await readFile(path.join(sampleDataDir, file));
  const base64 = bytes.toString("base64");
  const requestContext = new RequestContext<SearchImagesRequestContext>();
  requestContext.set("imageBase64", base64);
  requestContext.set("imageMimeType", mimeType);
  return {
    prompt:
      "A user has attached an image and wants to know which stored image it matches and what it shows.",
    requestContext,
  };
}

// ---------- cold-cache priming (run before Part A) ----------

// Checks 1-2 assert exact hit/miss counts against a *persistent* sled store, so
// without this step Run 1 is only genuinely cold on the very first invocation ever
// made against a given Zerocache instance -- any second run would see Run 1 as all
// hits and both checks would fail, not because anything broke but because the store
// already had the entries. DELETE is pure cache-key computation (no provider call,
// no billing -- see CLAUDE.md's API contract), and idempotent (deleting an
// already-absent key still succeeds), so wiping every item Parts A-C's ingestion
// could have created, from both v1 and v2 (they overlap on 7 of 8 items but differ on
// pricing.md, and v2 adds bulk-export-feature.md), makes the script safely
// re-runnable against a store in any prior state, not just a freshly wiped one.
async function primeColdCache(): Promise<void> {
  for (const dir of [SAMPLE_V1, SAMPLE_V2]) {
    const files = await readdir(dir);
    const textFiles = files.filter((f) => f.endsWith(".txt") || f.endsWith(".md"));
    const imageFiles = files.filter(
      (f) => f.endsWith(".png") || f.endsWith(".jpg") || f.endsWith(".jpeg"),
    );

    for (const file of textFiles) {
      const text = await readFile(path.join(dir, file), "utf-8");
      await deleteText({
        baseUrl: ZEROCACHE_BASE_URL,
        provider: "openai",
        apiKey: OPENAI_API_KEY,
        model: TEXT_MODEL,
        input: text,
      });
    }

    for (const file of imageFiles) {
      const bytes = await readFile(path.join(dir, file));
      const mimeType = file.endsWith(".png") ? "image/png" : "image/jpeg";
      await deleteImage({
        baseUrl: ZEROCACHE_BASE_URL,
        apiKey: GEMINI_API_KEY,
        model: IMAGE_MODEL,
        images: [{ mimeType, base64: bytes.toString("base64") }],
      });
    }
  }
}

// ---------- Part A: the cache-benefit story ----------

async function partA(): Promise<void> {
  console.log("=== Part A: the cache-benefit story ===");
  const rows: RunRow[] = [];

  await run("check-1-cold-ingest-v1", async () => {
    rows.push(
      await runIngest(
        "Check 1: Run 1 (cold ingest v1) -- 8 misses, 0 hits",
        "1. Cold ingest (v1)",
        SAMPLE_V1,
        {
          hits: 0,
          misses: 8,
        },
      ),
    );
  });

  await run("check-2-rebuild-v1-unchanged", async () => {
    rows.push(
      await runIngest(
        "Check 2: Run 2 (rebuild v1, unchanged) -- 8 hits, 0 misses",
        "2. Rebuild, unchanged",
        SAMPLE_V1,
        { hits: 8, misses: 0 },
        { expectZeroTokens: true },
      ),
    );
  });

  await run("check-3-rebuild-v2-edit-and-new", async () => {
    rows.push(
      await runIngest(
        "Check 3: Run 3 (rebuild v2, 1 edit + 1 new file) -- exactly 7 hits, 2 misses",
        "3. Rebuild, 1 edit+1 new",
        SAMPLE_V2,
        { hits: 7, misses: 2 },
      ),
    );
  });

  // Check 4: summary table for the human reader -- not itself an assertion (Items/
  // Hits/Misses are already asserted exactly above; Tokens billed/Duration vary by
  // real API response and are printed as evidence, not checked against a fixed value).
  printSummaryTable(rows);
}

// ---------- Part B: agentic RAG proof ----------

async function partB(): Promise<void> {
  console.log(
    "\n=== Part B: agentic RAG proof (tool-call trace assertions) ===",
  );

  await run("check-5-text-tool-selection", async () => {
    const trace: AgentTrace = await ragAgent.generate(
      "What's included in the Pro tier?",
    );

    const tools = calledTools(trace);
    const calledDocs = tools.includes("searchDocuments");
    const calledImages = tools.includes("searchImages");
    record(
      "Check 5: searchDocuments called, searchImages NOT called",
      calledDocs && !calledImages ? "PASS" : "FAIL",
      `tools called=${tools.join(",") || "(none)"}`,
    );

    const docCalls = toolResultPayloads<SearchDocumentsResult>(
      trace,
      "searchDocuments",
    );
    const files = new Set(
      docCalls.flatMap((r) =>
        r.results.map((it) => it.file).filter((f): f is string => !!f),
      ),
    );
    record(
      "Check 5: searchDocuments results include pricing.md",
      files.has("pricing.md") ? "PASS" : "FAIL",
      `files=${[...files].join(",")}`,
    );

    const text = trace.text;
    const reflectsV2 = text.includes("12") && text.includes("750");
    const reflectsStaleV1 =
      text.includes("9/month") ||
      text.includes("500GB") ||
      text.includes("500 GB");
    record(
      "Check 5: final answer reflects v2 pricing ($12/month, 750GB), not stale v1 numbers ($9/month, 500GB)",
      reflectsV2 && !reflectsStaleV1 ? "PASS" : "FAIL",
      `answer="${text.slice(0, 300)}"`,
    );
  });

  await run("check-6-image-tool-selection", async () => {
    const { prompt, requestContext } = await buildImageQuery(
      SAMPLE_V2,
      "architecture-diagram.png",
      "image/png",
    );
    const trace: AgentTrace = await ragAgent.generate(prompt, { requestContext });

    const tools = calledTools(trace);
    const calledImages = tools.includes("searchImages");
    const calledDocs = tools.includes("searchDocuments");
    record(
      "Check 6: searchImages called, searchDocuments NOT called",
      calledImages && !calledDocs ? "PASS" : "FAIL",
      `tools called=${tools.join(",") || "(none)"}`,
    );

    const imageCalls = toolResultPayloads<SearchImagesResult>(
      trace,
      "searchImages",
    );
    const entries = imageCalls.flatMap((r) => r.results);
    if (entries.length === 0) {
      record(
        "Check 6: top-scoring searchImages result is architecture-diagram.png, not dashboard-screenshot.png",
        "FAIL",
        "searchImages returned no results (or was never called)",
      );
    } else {
      const top = entries.reduce((best, cur) =>
        cur.score > best.score ? cur : best,
      );
      record(
        "Check 6: top-scoring searchImages result is architecture-diagram.png, not dashboard-screenshot.png",
        top.file === "architecture-diagram.png" ? "PASS" : "FAIL",
        `top=${JSON.stringify(top)}`,
      );
    }
  });

  await run("check-7-multi-hop-synthesis", async () => {
    const trace: AgentTrace = await ragAgent.generate(
      "How much does the Pro tier cost per month, and does that tier include Bulk Export?",
    );

    const tools = calledTools(trace);
    record(
      "Check 7: searchDocuments called at least once",
      tools.includes("searchDocuments") ? "PASS" : "FAIL",
      `tools called=${tools.join(",") || "(none)"}`,
    );

    const docCalls = toolResultPayloads<SearchDocumentsResult>(
      trace,
      "searchDocuments",
    );
    const files = new Set(
      docCalls.flatMap((r) =>
        r.results.map((it) => it.file).filter((f): f is string => !!f),
      ),
    );
    record(
      "Check 7: union of all searchDocuments results includes pricing.md AND bulk-export-feature.md",
      files.has("pricing.md") && files.has("bulk-export-feature.md")
        ? "PASS"
        : "FAIL",
      `files=${[...files].join(",")}`,
    );

    const lower = trace.text.toLowerCase();
    const hasPrice = lower.includes("12");
    const statesIncluded =
      lower.includes("bulk export") &&
      (lower.includes("include") || lower.includes("yes"));
    record(
      "Check 7: final answer states the $12/month price AND that Bulk Export is included on the Pro tier",
      hasPrice && statesIncluded ? "PASS" : "FAIL",
      `answer="${trace.text.slice(0, 400)}"`,
    );
  });

  await run("check-8-judgment-not-to-retrieve", async () => {
    const trace: AgentTrace = await ragAgent.generate(
      "What is the capital of France?",
    );

    const tools = calledTools(trace);
    record(
      "Check 8: neither searchDocuments nor searchImages called",
      tools.length === 0 ? "PASS" : "FAIL",
      `tools called=${tools.join(",") || "(none)"}`,
    );

    // Checking for the literal phrase "knowledge base" here would false-positive on the
    // correct, desired decline response ("That question is outside the scope of this
    // knowledge base") -- saying that phrase while declining is not hallucinated
    // grounding. What actually matters is whether the agent fabricated Aurora-specific
    // facts it could not have known without a tool call it never made.
    const lower = trace.text.toLowerCase();
    const fabricatedFacts = [
      "$9",
      "$12",
      "pro tier",
      "free tier",
      "business tier",
      "bulk export",
      "500gb",
      "750gb",
    ];
    const fabricatesAuroraContent = fabricatedFacts.some((needle) =>
      lower.includes(needle),
    );
    record(
      "Check 8: final answer doesn't fabricate Aurora-specific content",
      !fabricatesAuroraContent ? "PASS" : "FAIL",
      `answer="${trace.text.slice(0, 300)}"`,
    );
  });
}

// ---------- Part C: delete, HTTP-level ----------

async function partC(): Promise<void> {
  console.log(
    "\n=== Part C: delete, HTTP-level (Task 5's delete routes, only exercised end-to-end here) ===",
  );

  await run("check-9-10-delete-then-reembed-text", async () => {
    // v1 and v2's getting-started.md are byte-identical (Task 10) -- reading from
    // either path derives the same CacheKey; v2 is used since it's the current/final
    // ingested state after Run 3.
    const textContent = await readFile(
      path.join(SAMPLE_V2, "getting-started.md"),
      "utf-8",
    );

    const deleteResult = await deleteText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey: OPENAI_API_KEY,
      model: TEXT_MODEL,
      input: textContent,
    });
    record(
      "Check 9: deleteText(getting-started.md) -> deleted: 1",
      deleteResult.deleted === 1 ? "PASS" : "FAIL",
      `deleted=${deleteResult.deleted}`,
    );

    const reembed = await embedText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey: OPENAI_API_KEY,
      model: TEXT_MODEL,
      input: textContent,
    });
    record(
      "Check 10: re-embedText(getting-started.md) is a miss, not a hit -- proves delete actually removed the entry",
      reembed.misses === 1 && reembed.hits === 0 ? "PASS" : "FAIL",
      `hits=${reembed.hits} misses=${reembed.misses}`,
    );
  });

  await run("check-11-delete-then-reembed-image", async () => {
    const imageBytes = await readFile(
      path.join(SAMPLE_V2, "architecture-diagram.png"),
    );
    const base64 = imageBytes.toString("base64");
    const images = [{ mimeType: "image/png", base64 }];

    const deleteResult = await deleteImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey: GEMINI_API_KEY,
      model: IMAGE_MODEL,
      images,
    });
    record(
      "Check 11a: deleteImage(architecture-diagram.png) -> deleted: 1",
      deleteResult.deleted === 1 ? "PASS" : "FAIL",
      `deleted=${deleteResult.deleted}`,
    );

    const reembed = await embedImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey: GEMINI_API_KEY,
      model: IMAGE_MODEL,
      images,
    });
    record(
      "Check 11b: re-embedImage(architecture-diagram.png) is a miss, not a hit -- proves delete actually removed the entry",
      reembed.misses === 1 && reembed.hits === 0 ? "PASS" : "FAIL",
      `hits=${reembed.hits} misses=${reembed.misses}`,
    );
  });

  // Check 12: restore state. No separate action needed -- checks 10 and 11's re-embed
  // calls above already repopulate the cache for getting-started.md and
  // architecture-diagram.png as a side effect of proving the miss, so the KB is back
  // to its post-Run-3 state. Part C is deliberately run after Parts A and B (not
  // interleaved) specifically so a failure here can never invalidate the agent-query
  // assertions in checks 5-8, which already ran and had their results recorded above.
}

async function main() {
  // Throwaway request so the process's very first HTTP connection isn't the one
  // measured for Run 1's duration -- matches the warm-up pattern in
  // demo/langchain-ts/src/battle-test.ts and demo/llamaindex-python/src/battle_test.py.
  await fetch(`${ZEROCACHE_BASE_URL}/health`).catch(() => {});

  console.log("Priming a cold cache (deleting any pre-existing v1/v2 entries)...");
  await primeColdCache();

  await partA();
  await partB();
  await partC();

  console.log("\n=== summary ===");
  const counts: Record<Result, number> = { PASS: 0, FAIL: 0 };
  for (const r of results) counts[r.result]++;
  console.log(`${counts.PASS} passed, ${counts.FAIL} failed`);

  if (counts.FAIL > 0) {
    console.log("\nFAILURES:");
    for (const r of results.filter((r) => r.result === "FAIL"))
      console.log(`  - ${r.name}: ${r.detail ?? ""}`);
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("battle-test harness crashed:", err);
  process.exitCode = 1;
});

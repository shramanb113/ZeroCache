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

// Sums every line for a given metric family whose label set contains all of `filters`.
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
// via requestContext, not as base64 text in the prompt.
async function buildImageQuery(
  sampleDataDir: string,
  file: string,
  mimeType: string,
): Promise<{
  prompt: string;
  requestContext: RequestContext<SearchImagesRequestContext>;
}> {
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

// Deletes every cache entry Parts A-C's ingestion could create (v1 + v2, text +
// images) so Checks 1-2's exact cold hit/miss counts hold on every re-run.
async function primeColdCache(): Promise<void> {
  for (const dir of [SAMPLE_V1, SAMPLE_V2]) {
    const files = await readdir(dir);
    const textFiles = files.filter(
      (f) => f.endsWith(".txt") || f.endsWith(".md"),
    );
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
    const trace: AgentTrace = await ragAgent.generate(prompt, {
      requestContext,
    });

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
}

async function main() {
  await fetch(`${ZEROCACHE_BASE_URL}/health`).catch(() => {});

  console.log(
    "Priming a cold cache (deleting any pre-existing v1/v2 entries)...",
  );
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

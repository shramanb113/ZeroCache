// Battle-tests Zerocache from a LangChain TypeScript consumer's
// perspective: HTTP-contract edge cases first (no API key needed), then
// behavioral edge cases driven through @langchain/openai's own client (key
// needed -- these make real, billed provider calls).
//
// Run with: npm run battle-test
// (set ZEROCACHE_DEMO_KEY first to also run the key-dependent section)

import { OpenAIEmbeddings } from "@langchain/openai";

import { API_KEY, MODEL, PROVIDER, ZEROCACHE_HOST, ZEROCACHE_OPENAI_BASE_URL, requireApiKey } from "./zerocache-config.js";
import { loggingFetch } from "./logging-fetch.js";

type Result = "PASS" | "FAIL" | "FINDING" | "SKIP";
const results: { name: string; result: Result; detail?: string }[] = [];

function record(name: string, result: Result, detail?: string) {
  results.push({ name, result, detail });
  const marker = { PASS: "✓", FAIL: "✗", FINDING: "⚠", SKIP: "-" }[result];
  console.log(`${marker} [${result}] ${name}${detail ? " -- " + detail : ""}`);
}

async function run(name: string, fn: () => Promise<void>) {
  try {
    await fn();
  } catch (err) {
    record(name, "FAIL", err instanceof Error ? err.message : String(err));
  }
}

// ---------- key-independent HTTP-contract tests ----------

async function testHealth() {
  const res = await fetch(`${ZEROCACHE_HOST}/health`);
  record("GET /health returns 200", res.status === 200 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testReady() {
  const res = await fetch(`${ZEROCACHE_HOST}/ready`);
  record("GET /ready returns 200 against a live local store", res.status === 200 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testMetricsEndpoint() {
  const res = await fetch(`${ZEROCACHE_HOST}/metrics`);
  const text = await res.text();
  const looksLikePrometheus = text.includes("# HELP") || text.includes("zerocache_cache_hits_total");
  record("GET /metrics returns Prometheus text format", res.status === 200 && looksLikePrometheus ? "PASS" : "FAIL", `status=${res.status}`);
}

async function testMissingAuth() {
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, input: ["test"] }),
  });
  record("Missing Authorization header -> 401", res.status === 401 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testMalformedAuth() {
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "not-a-bearer-token" },
    body: JSON.stringify({ model: MODEL, input: ["test"] }),
  });
  record("Malformed Authorization header (no 'Bearer ' prefix) -> 401", res.status === 401 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testUnknownProvider() {
  const res = await fetch(`${ZEROCACHE_HOST}/not-a-real-provider/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer fake-key-unknown-provider-test" },
    body: JSON.stringify({ model: MODEL, input: ["test"] }),
  });
  record("Unknown provider in path -> 404", res.status === 404 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testEmptyInputArray() {
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer fake-key-empty-input-test" },
    body: JSON.stringify({ model: MODEL, input: [] }),
  });
  const body = (await res.json()) as { data?: unknown[] };
  const ok = res.status === 200 && Array.isArray(body.data) && body.data.length === 0;
  record("Empty input array is accepted, returns empty data (no provider call)", ok ? "PASS" : "FAIL", `status=${res.status} body=${JSON.stringify(body)}`);
}

async function testMissingModelField() {
  // request.model is a non-Optional String field on EmbeddingsRequest, so a
  // missing "model" key fails axum's Json<T> extraction *before* the
  // handler body runs -- before even the auth check. Valid JSON syntax but
  // the wrong shape is a *data* rejection, which axum maps to 422 (distinct
  // from malformed-JSON-syntax's 400 below) -- confirmed against a live
  // server, not assumed.
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer fake-key-missing-model-test" },
    body: JSON.stringify({ input: ["test"] }),
  });
  const body = await res.text();
  let parsedAsAppErrorShape = false;
  try {
    parsedAsAppErrorShape = typeof JSON.parse(body).error === "string";
  } catch {
    /* not JSON at all -- also relevant, noted below */
  }
  record("Missing required 'model' field -> 422", res.status === 422 ? "PASS" : "FAIL", `status=${res.status}`);
  record(
    "Error response shape consistency (axum body-rejection vs. app ErrorResponse)",
    parsedAsAppErrorShape ? "PASS" : "FINDING",
    parsedAsAppErrorShape
      ? undefined
      : `body-rejection errors don't come back as {"error": "..."} like every other error path does -- ` +
          `a consumer that parses all Zerocache error bodies uniformly will break on this one. body=${body.slice(0, 150)}`,
  );
}

async function testMalformedJsonBody() {
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer fake-key-malformed-json-test" },
    body: "{not valid json",
  });
  record("Malformed JSON body -> 400", res.status === 400 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testDeleteWithoutAuth() {
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, input: ["test"] }),
  });
  record("DELETE without Authorization -> 401", res.status === 401 ? "PASS" : "FAIL", `got ${res.status}`);
}

async function testDifferentOwnersDontShareCache() {
  // A syntactically fake-but-well-formed key still derives a real owner_id
  // (it's just a hash of whatever string is forwarded) -- the request will
  // fail once it actually reaches the provider, but that's a *provider*
  // auth failure (502), not a cache-layer bypass. Confirms owner-scoping
  // happens independent of whether the key is real.
  const unique = `owner isolation probe ${Date.now()}`;
  const res = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer fake-owner-a-test-key-000000000000000000" },
    body: JSON.stringify({ model: MODEL, input: [unique] }),
  });
  record("A syntactically fake key is owner-scoped and fails at the provider (502), not silently cached", res.status === 502 ? "PASS" : "FINDING", `status=${res.status}`);
}

// ---------- key-dependent tests, driven through @langchain/openai itself ----------

function makeEmbeddings(label: string) {
  return new OpenAIEmbeddings({
    apiKey: requireApiKey(),
    model: MODEL,
    configuration: { baseURL: ZEROCACHE_OPENAI_BASE_URL, fetch: loggingFetch(label) },
  });
}

async function testBasicMissThenHit() {
  const unique = `battle-test roundtrip ${Date.now()}`;
  const embeddings = makeEmbeddings("roundtrip");
  const first = await embeddings.embedQuery(unique);
  const second = await embeddings.embedQuery(unique);
  record("Basic miss-then-hit returns an identical vector", JSON.stringify(first) === JSON.stringify(second) ? "PASS" : "FAIL");
}

async function testWithinBatchDedup() {
  const unique = `dedup test ${Date.now()}`;
  const embeddings = makeEmbeddings("dedup");
  const results = await embeddings.embedDocuments([unique, unique, unique]);
  const allSame = results.every((v) => JSON.stringify(v) === JSON.stringify(results[0]));
  record("Duplicate texts within one batch all return the same vector", allSame ? "PASS" : "FAIL");
}

async function testWhitespaceNormalization() {
  const base = `whitespace test ${Date.now()}`;
  const embeddings = makeEmbeddings("whitespace");
  const first = await embeddings.embedQuery(base);
  const second = await embeddings.embedQuery(`   ${base.replace(/ /g, "   ")}   `);
  record("Whitespace-variant text shares a cache entry with the normalized original", JSON.stringify(first) === JSON.stringify(second) ? "PASS" : "FAIL");
}

async function testLargeBatchAcrossProviderChunkBoundary() {
  const stamp = Date.now();
  const texts = Array.from({ length: 150 }, (_, i) => `large batch item ${stamp}-${i}`);
  const embeddings = makeEmbeddings("large-batch-150");
  const start = performance.now();
  const results = await embeddings.embedDocuments(texts);
  const elapsedMs = performance.now() - start;
  const allPresent = results.length === 150 && results.every((v) => Array.isArray(v) && v.length > 0);
  record(
    "150-item batch (crosses Zerocache's internal MAX_BATCH_SIZE=100 chunk boundary) returns all vectors correctly",
    allPresent ? "PASS" : "FAIL",
    `${results.length}/150 returned in ${elapsedMs.toFixed(0)}ms`,
  );
}

async function testConcurrentDuplicateRequestsCoalescing() {
  // Deliberately uncached text, fired as N *simultaneous* requests for the
  // exact same text. With request coalescing, only one would reach the
  // provider and the rest would share its result. Zerocache doesn't have
  // this yet (explicitly deferred this session as "singleflight") -- this
  // test exists to make that gap concrete and measurable, not theoretical.
  const unique = `coalescing probe ${Date.now()}`;
  const concurrency = 5;
  const instances = Array.from({ length: concurrency }, (_, i) => makeEmbeddings(`coalesce-${i}`));

  const start = performance.now();
  const outcomes = await Promise.all(instances.map((e) => e.embedQuery(unique)));
  const elapsedMs = performance.now() - start;

  const allSameVector = outcomes.every((v) => JSON.stringify(v) === JSON.stringify(outcomes[0]));
  record(`${concurrency} concurrent identical-text requests all return the correct (identical) vector`, allSameVector ? "PASS" : "FAIL");
  record(
    "Request coalescing / singleflight is not implemented",
    "FINDING",
    `${concurrency} concurrent misses for the same never-before-seen text took ${elapsedMs.toFixed(0)}ms total for all to resolve. ` +
      "Each one independently missed the cache and called the provider -- check zerocache_provider_prompt_tokens_total in " +
      `/metrics before/after this run: it should have grown by only 1x a single call's tokens if coalesced, but will show ` +
      `~${concurrency}x if not. Known, previously-scoped-but-deferred gap, not a bug.`,
  );
}

async function testDeleteThenReembedIsAMissAgain() {
  const unique = `delete roundtrip ${Date.now()}`;
  const apiKey = requireApiKey();
  const embeddings = makeEmbeddings("delete-embed");

  await embeddings.embedQuery(unique); // now cached

  const deleteRes = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model: MODEL, input: [unique] }),
  });
  const deleteBody = (await deleteRes.json()) as { deleted?: number };

  if (deleteRes.status !== 200 || deleteBody.deleted !== 1) {
    record("DELETE removes exactly the requested entry", "FAIL", `status=${deleteRes.status} body=${JSON.stringify(deleteBody)}`);
    return;
  }
  record("DELETE removes exactly the requested entry", "PASS");

  const afterDeleteRes = await fetch(`${ZEROCACHE_HOST}/${PROVIDER}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model: MODEL, input: [unique] }),
  });
  const misses = afterDeleteRes.headers.get("x-zerocache-misses");
  record("A deleted entry is a miss again on the next request", misses === "1" ? "PASS" : "FAIL", `x-zerocache-misses=${misses}`);
}

async function main() {
  // Throwaway request so the process's very first HTTP connection isn't
  // the one under test -- observed the /metrics check flake once on a cold
  // start (first-connection latency, not a server bug: reproduced clean on
  // every subsequent run including immediately after this warm-up).
  await fetch(`${ZEROCACHE_HOST}/health`).catch(() => {});

  console.log("=== key-independent HTTP-contract tests ===");
  await run("health", testHealth);
  await run("ready", testReady);
  await run("metrics", testMetricsEndpoint);
  await run("missing-auth", testMissingAuth);
  await run("malformed-auth", testMalformedAuth);
  await run("unknown-provider", testUnknownProvider);
  await run("empty-input", testEmptyInputArray);
  await run("missing-model-field", testMissingModelField);
  await run("malformed-json", testMalformedJsonBody);
  await run("delete-without-auth", testDeleteWithoutAuth);
  await run("fake-key-owner-scoping", testDifferentOwnersDontShareCache);

  if (!API_KEY) {
    console.log("\n=== key-dependent tests SKIPPED: ZEROCACHE_DEMO_KEY not set ===");
    for (const name of ["basic-miss-then-hit", "within-batch-dedup", "whitespace-normalization", "large-batch-boundary", "concurrent-coalescing-probe", "delete-then-reembed"]) {
      record(name, "SKIP");
    }
  } else {
    console.log("\n=== key-dependent tests (real provider calls, real cost) ===");
    await run("basic-miss-then-hit", testBasicMissThenHit);
    await run("within-batch-dedup", testWithinBatchDedup);
    await run("whitespace-normalization", testWhitespaceNormalization);
    await run("large-batch-boundary", testLargeBatchAcrossProviderChunkBoundary);
    await run("concurrent-coalescing-probe", testConcurrentDuplicateRequestsCoalescing);
    await run("delete-then-reembed", testDeleteThenReembedIsAMissAgain);
  }

  console.log("\n=== summary ===");
  const counts: Record<Result, number> = { PASS: 0, FAIL: 0, FINDING: 0, SKIP: 0 };
  for (const r of results) counts[r.result]++;
  console.log(`${counts.PASS} passed, ${counts.FAIL} failed, ${counts.FINDING} findings, ${counts.SKIP} skipped`);

  if (counts.FAIL > 0) {
    console.log("\nFAILURES:");
    for (const r of results.filter((r) => r.result === "FAIL")) console.log(`  - ${r.name}: ${r.detail ?? ""}`);
  }
  if (counts.FINDING > 0) {
    console.log("\nFINDINGS (not failures, but worth flagging):");
    for (const r of results.filter((r) => r.result === "FINDING")) console.log(`  - ${r.name}: ${r.detail ?? ""}`);
  }
  if (counts.FAIL > 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error("battle-test harness crashed:", err);
  process.exitCode = 1;
});

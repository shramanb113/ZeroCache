// End-to-end battle test for the semantic LLM completion cache (CLAUDE.md
// Deviations item 21). Proves the thing the feature is sold on: an AI agent
// that runs the same task more than once, and the short auxiliary LLM calls
// agents make in bulk, get served from cache the second time -- 100% off both
// input and output tokens for every hit.
//
// This is the completion-cache counterpart to demo/mastra/battle-test.ts, and
// uses the same measurement technique: snapshot GET /metrics before and after
// a run and diff the counters. It has no framework dependency -- just global
// fetch and a hand-rolled assert -- so it runs with plain `node`, no install.
//
// Requires a real running Zerocache instance (ZEROCACHE_BASE_URL, default
// http://localhost:8080) and a real OPENAI_API_KEY. It makes real, billed
// OpenAI chat calls on the FIRST run of each phase; the point of the test is
// that the second run bills nothing.
//
//   OPENAI_API_KEY=sk-... node demo/completion-cache/battle-test.mjs
//
// Any built-in chat provider works with no server config; set
// ZEROCACHE_CHAT_PROVIDER to pick which one this script calls. To run it
// against Gemini's OpenAI-compat endpoint:
//
//   ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" cargo run -p zerocache-http
//   OPENAI_API_KEY=<gemini key> MODEL=gemini-3.5-flash-lite ZEROCACHE_CHAT_PROVIDER=gemini node demo/completion-cache/battle-test.mjs
//
// Safe to re-run: every invocation folds a fresh random runId into the system
// prompt, so each run starts from a cold cache for its own keys without
// needing a DELETE route (the completion cache has none in v1). Re-running
// therefore does re-bill the cold phase each time.
//
// What it does NOT test: the deferred semantic-similarity tier (this is
// exact-match only), streaming, or tool-call execution. The "agent loop" here
// is a scripted stand-in -- a fixed multi-turn sequence where each turn's
// request is built from the previous turn's actual (possibly cached) response,
// so the whole dependent chain stays on the hit path on a repeat run, which is
// exactly the "repeated agent run" money case.

const BASE_URL = (process.env.ZEROCACHE_BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.MODEL || "gpt-4o-mini";
const PROVIDER = process.env.ZEROCACHE_CHAT_PROVIDER || "openai";

// Illustrative list prices, USD per 1M tokens. Not authoritative -- override
// for your model/tier. Used only to turn the tokens-saved counters into a
// dollar figure at the end.
const PRICE_PER_MTOK = {
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-4o": { input: 2.5, output: 10.0 },
};

if (!API_KEY) {
  console.error("OPENAI_API_KEY is required (this test makes real billed calls on the cold run).");
  process.exit(2);
}

let failures = 0;
function check(label, cond, detail) {
  const ok = Boolean(cond);
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` -- ${detail}` : ""}`);
  if (!ok) failures += 1;
}

async function metricsSnapshot() {
  const res = await fetch(`${BASE_URL}/metrics`);
  if (!res.ok) throw new Error(`GET /metrics -> ${res.status}`);
  const text = await res.text();
  const sumOf = (name) => {
    let total = 0;
    for (const line of text.split("\n")) {
      if (line.startsWith("#") || !line.startsWith(name)) continue;
      const value = Number(line.slice(line.lastIndexOf(" ") + 1));
      if (Number.isFinite(value)) total += value;
    }
    return total;
  };
  return {
    hits: sumOf("zerocache_completion_cache_hits_total"),
    misses: sumOf("zerocache_completion_cache_misses_total"),
    promptSaved: sumOf("zerocache_completion_prompt_tokens_saved_total"),
    completionSaved: sumOf("zerocache_completion_completion_tokens_saved_total"),
  };
}

function delta(before, after) {
  return {
    hits: after.hits - before.hits,
    misses: after.misses - before.misses,
    promptSaved: after.promptSaved - before.promptSaved,
    completionSaved: after.completionSaved - before.completionSaved,
  };
}

async function chat(messages) {
  const res = await fetch(`${BASE_URL}/${PROVIDER}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({ model: MODEL, messages, temperature: 0 }),
  });
  const json = await res.json();
  if (!res.ok) {
    throw new Error(`chat -> ${res.status}: ${JSON.stringify(json).slice(0, 300)}`);
  }
  return {
    content: json.choices?.[0]?.message?.content ?? "",
    hit: res.headers.get("x-zerocache-completion-hit") === "true",
  };
}

// A scripted stand-in for an agent working a ticket: system prompt with a
// (fake) tool contract, then a dependent multi-turn chain. Each turn's prompt
// includes the previous turn's actual response text, so on a repeat run the
// whole chain only stays identical because the cached replays are byte-exact.
async function runAgentLoop(runId) {
  const system = {
    role: "system",
    content:
      `You are a support triage agent. Session ${runId}. ` +
      `Answer in one short sentence. When asked to pick a tool, reply with exactly one of: ` +
      `search_docs, escalate, reply_directly.`,
  };
  const transcript = [system];
  const turns = [
    "Ticket: 'I was charged twice for my subscription this month.' Classify the issue in 3 words.",
    "Given that classification, which tool should you call next?",
    "Draft the first line of the customer reply.",
    "Summarize what you did in this ticket in one sentence.",
  ];
  const hits = [];
  for (const turn of turns) {
    transcript.push({ role: "user", content: turn });
    const { content, hit } = await chat(transcript);
    transcript.push({ role: "assistant", content });
    hits.push(hit);
  }
  return hits;
}

async function main() {
  console.log(`Zerocache: ${BASE_URL}   model: ${MODEL}\n`);

  const runId = Math.random().toString(36).slice(2, 10);

  // ---- Phase 1: an agent runs a task twice ----
  console.log("Phase 1 -- the same agent loop, run twice");

  const p1a = await metricsSnapshot();
  const coldHits = await runAgentLoop(runId);
  const p1b = await metricsSnapshot();
  const cold = delta(p1a, p1b);
  console.log(
    `  cold run:   ${cold.misses} misses, ${cold.hits} hits ` +
      `(${coldHits.filter(Boolean).length}/${coldHits.length} turns from cache)`,
  );

  const warmHits = await runAgentLoop(runId);
  const p1c = await metricsSnapshot();
  const warm = delta(p1b, p1c);
  console.log(
    `  repeat run: ${warm.misses} misses, ${warm.hits} hits ` +
      `(${warmHits.filter(Boolean).length}/${warmHits.length} turns from cache)`,
  );
  console.log(
    `  tokens the repeat run did NOT pay for: ` +
      `${warm.promptSaved} prompt + ${warm.completionSaved} completion`,
  );

  check("cold run is all misses", cold.misses === 4 && cold.hits === 0);
  check("repeat run is all hits", warm.hits === 4 && warm.misses === 0, `hits=${warm.hits} misses=${warm.misses}`);
  check("every turn of the repeat run came from cache", warmHits.every(Boolean));
  check(
    "repeat run saved real tokens",
    warm.promptSaved > 0 && warm.completionSaved > 0,
    `${warm.promptSaved}+${warm.completionSaved}`,
  );

  // ---- Phase 2: a bulk auxiliary call ----
  // Agents fire the same short classify/extract/summarize prompt over and
  // over. Here: one prompt sent 8 times. First is a miss, the rest are free.
  console.log("\nPhase 2 -- one auxiliary prompt, sent 8 times");
  const auxPrompt = [
    { role: "system", content: `Classify log lines as INFO, WARN, or ERROR. Session ${runId}. One word only.` },
    { role: "user", content: "log: connection reset by peer while streaming response body" },
  ];
  const p2a = await metricsSnapshot();
  const auxHits = [];
  for (let i = 0; i < 8; i += 1) auxHits.push((await chat(auxPrompt)).hit);
  const p2b = await metricsSnapshot();
  const aux = delta(p2a, p2b);
  console.log(`  ${aux.misses} miss + ${aux.hits} hits over 8 identical calls`);
  check("8 identical auxiliary calls = 1 miss + 7 hits", aux.misses === 1 && aux.hits === 7);

  // ---- Money ----
  const totalPromptSaved = warm.promptSaved + aux.promptSaved;
  const totalCompletionSaved = warm.completionSaved + aux.completionSaved;
  const price = PRICE_PER_MTOK[MODEL];
  console.log("\nTokens saved this run (repeat agent loop + bulk auxiliary):");
  console.log(`  prompt:     ${totalPromptSaved}`);
  console.log(`  completion: ${totalCompletionSaved}`);
  if (price) {
    const usd =
      (totalPromptSaved / 1e6) * price.input + (totalCompletionSaved / 1e6) * price.output;
    console.log(`  ~$${usd.toFixed(6)} at ${MODEL} list price (illustrative)`);
    console.log(
      `  an agent fleet re-running this shape 10k times/day would save ` +
        `~$${(usd * 10000).toFixed(2)}/day`,
    );
  }

  console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

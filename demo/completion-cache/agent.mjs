// A real tool-calling agent, run through Zerocache's completion cache.
//
// Unlike battle-test.mjs (a scripted multi-turn chain), this is an actual
// ReAct-style agent loop: the model decides which tool to call, we execute
// it, feed the result back, and repeat until the model produces a final
// answer. Every step is a real OpenAI-shaped /v1/chat/completions call with
// a `tools` array, sent at temperature 0 so it is deterministic and
// therefore cacheable.
//
// The scenario: a support-triage agent with three tools over in-script data
// (a tiny KB and an orders table). It resolves support tickets.
//
// The money case being demonstrated is a CI / eval loop: you run the same
// suite of tickets through the agent on every commit. The first run pays
// full price; every run after that is served entirely from the completion
// cache -- 100% off input AND output tokens, including all the intermediate
// tool-call turns -- until a prompt, a tool definition, or the model changes.
//
//   # terminal 1 -- register Gemini as a chat provider (no doubled /v1)
//   ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" \
//     cargo run -p zerocache-http
//
//   # terminal 2
//   OPENAI_API_KEY=<key> MODEL=gemini-3.5-flash-lite ZEROCACHE_CHAT_PROVIDER=gemini \
//     node demo/completion-cache/agent.mjs
//
// With no MODEL set it defaults to gpt-4o-mini and ZEROCACHE_CHAT_PROVIDER to
// openai, expecting a real OpenAI key. Any built-in chat provider works.
//
// Dependency-free: global fetch + a hand-rolled assert. Safe to re-run --
// each invocation folds a fresh runId into the system prompt, so it starts
// from a cold cache for its own keys (the completion cache has no DELETE in
// v1) and re-bills the cold run every time.

const BASE_URL = (process.env.ZEROCACHE_BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.MODEL || "gpt-4o-mini";
const PROVIDER = process.env.ZEROCACHE_CHAT_PROVIDER || "openai";
const MAX_STEPS = 6;

// Illustrative list prices, USD per 1M tokens. Override for your model/tier.
const PRICE_PER_MTOK = {
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-4o": { input: 2.5, output: 10.0 },
  "gemini-2.5-flash": { input: 0.3, output: 2.5 },
  "gemini-3.5-flash-lite": { input: 0.1, output: 0.4 },
};

if (!API_KEY) {
  console.error("OPENAI_API_KEY is required (the cold run makes real billed calls).");
  process.exit(2);
}

let failures = 0;
function check(label, cond, detail) {
  const ok = Boolean(cond);
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` -- ${detail}` : ""}`);
  if (!ok) failures += 1;
}

// ---------------------------------------------------------------------------
// The agent's world: a tiny knowledge base and an orders table.
// ---------------------------------------------------------------------------

const KB = [
  {
    id: "kb-refund-window",
    title: "Refund window",
    body: "Refunds are available within 30 days of the delivery date. After 30 days, only store credit is offered.",
  },
  {
    id: "kb-double-charge",
    title: "Duplicate charges",
    body: "A duplicate charge is usually a pending authorization that drops off in 3-5 business days. If both charges have settled, issue a refund for the duplicate and tag the ticket 'billing'.",
  },
  {
    id: "kb-damaged",
    title: "Damaged on arrival",
    body: "For items damaged in transit, send a prepaid return label and ship a replacement immediately. No need to wait for the return.",
  },
  {
    id: "kb-address-change",
    title: "Address changes",
    body: "An order's shipping address can be changed only while its status is 'processing'. Once 'shipped', the customer must refuse delivery or return the package.",
  },
];

const ORDERS = {
  "A1001": { status: "shipped", total: 79.99, delivered_days_ago: 2, charges: 2, item: "Desk lamp" },
  "A1002": { status: "processing", total: 129.5, delivered_days_ago: null, charges: 1, item: "Office chair" },
  "A1003": { status: "delivered", total: 42.0, delivered_days_ago: 45, charges: 1, item: "Notebook set" },
};

const TOOLS = [
  {
    type: "function",
    function: {
      name: "search_kb",
      description: "Search the support knowledge base. Returns matching articles.",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "keywords to search for" } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_order",
      description: "Look up an order by its ID (e.g. A1001). Returns status, total, charge count, and delivery age.",
      parameters: {
        type: "object",
        properties: { order_id: { type: "string" } },
        required: ["order_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "resolve_ticket",
      description: "Close the ticket with a customer-facing resolution and an internal tag.",
      parameters: {
        type: "object",
        properties: {
          resolution: { type: "string", description: "one or two sentences for the customer" },
          tag: { type: "string", enum: ["billing", "shipping", "refund", "store-credit", "replacement"] },
        },
        required: ["resolution", "tag"],
      },
    },
  },
];

function runTool(name, args) {
  if (name === "search_kb") {
    const q = String(args.query || "").toLowerCase();
    const hits = KB.filter(
      (a) =>
        q.split(/\s+/).some((w) => w && (a.title.toLowerCase().includes(w) || a.body.toLowerCase().includes(w))),
    );
    return { articles: hits.length ? hits : KB.slice(0, 2) };
  }
  if (name === "get_order") {
    const o = ORDERS[String(args.order_id || "").toUpperCase()];
    return o ? { order_id: String(args.order_id).toUpperCase(), ...o } : { error: "no such order" };
  }
  if (name === "resolve_ticket") {
    return { closed: true, resolution: args.resolution, tag: args.tag };
  }
  return { error: `unknown tool ${name}` };
}

// ---------------------------------------------------------------------------
// The agent loop.
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Only the cold run reaches the provider, so it is the only place a
// provider-side rate limit (e.g. a free-tier per-minute quota) can bite.
// The repeat run is served entirely from Zerocache and never waits.
async function chat(messages) {
  const payload = JSON.stringify({ model: MODEL, messages, tools: TOOLS, temperature: 0 });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${BASE_URL}/${PROVIDER}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${API_KEY}` },
      body: payload,
    });
    const json = await res.json();
    if (res.ok) {
      return {
        message: json.choices?.[0]?.message ?? {},
        hit: res.headers.get("x-zerocache-completion-hit") === "true",
      };
    }
    const errText = JSON.stringify(json);
    // A per-DAY quota will not clear by waiting a minute -- fail fast rather
    // than burn retries (each retry is itself a billed request).
    const perDayQuota = /PerDay|RequestsPerDay/i.test(errText);
    if (res.status === 429 && attempt < 5 && !perDayQuota) {
      const wait = Math.min(60000, 8000 * 2 ** attempt);
      console.log(`  (provider rate-limited; waiting ${wait / 1000}s then retrying)`);
      await sleep(wait);
      continue;
    }
    throw new Error(`chat -> ${res.status}: ${errText.slice(0, 400)}`);
  }
}

// Pace only the cold run so a provider free-tier per-minute quota does not
// trip. A cached call returns instantly and needs no pacing.
const PACE_MS = Number(process.env.PACE_MS || 7000);

async function runAgent(runId, ticket) {
  const messages = [
    {
      role: "system",
      content:
        `You are a support-triage agent (session ${runId}). Resolve the ticket. ` +
        `Use search_kb to find policy, get_order to look up order facts, then call ` +
        `resolve_ticket exactly once with a short customer-facing resolution and a tag. ` +
        `Do not ask the user questions; act on the information available.`,
    },
    { role: "user", content: ticket },
  ];

  const stepHits = [];
  let toolCalls = 0;
  let resolution = null;

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const { message, hit } = await chat(messages);
    stepHits.push(hit);
    messages.push(message);
    if (!hit) await sleep(PACE_MS);

    const calls = message.tool_calls || [];
    if (calls.length === 0) break;

    for (const call of calls) {
      toolCalls += 1;
      let args = {};
      try {
        args = JSON.parse(call.function.arguments || "{}");
      } catch {
        args = {};
      }
      const result = runTool(call.function.name, args);
      if (call.function.name === "resolve_ticket") resolution = result;
      messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
    }
    if (resolution) {
      // Give the model one turn to acknowledge, then stop.
      const { message: final, hit: finalHit } = await chat(messages);
      stepHits.push(finalHit);
      messages.push(final);
      if (!finalHit) await sleep(PACE_MS);
      break;
    }
  }

  return { stepHits, toolCalls, resolution };
}

// ---------------------------------------------------------------------------
// Metrics diffing (same technique as demo/mastra/battle-test.ts).
// ---------------------------------------------------------------------------

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

function delta(a, b) {
  return {
    hits: b.hits - a.hits,
    misses: b.misses - a.misses,
    promptSaved: b.promptSaved - a.promptSaved,
    completionSaved: b.completionSaved - a.completionSaved,
  };
}

// ---------------------------------------------------------------------------

const TICKETS = [
  "Ticket A1001: 'I see TWO charges of $79.99 for my desk lamp order A1001, delivered 2 days ago. Fix this.'",
  "Ticket A1002: 'I need to change the shipping address on order A1002 (office chair) before it goes out.'",
  "Ticket A1003: 'My notebook set (order A1003) arrived 45 days ago and I want a full refund.'",
];

async function runSuite(runId, label) {
  const before = await metricsSnapshot();
  const results = [];
  for (const ticket of TICKETS) results.push(await runAgent(runId, ticket));
  const after = await metricsSnapshot();
  const d = delta(before, after);
  const totalSteps = results.reduce((n, r) => n + r.stepHits.length, 0);
  const cachedSteps = results.reduce((n, r) => n + r.stepHits.filter(Boolean).length, 0);
  console.log(
    `  ${label}: ${d.misses} misses, ${d.hits} hits across ${totalSteps} model calls ` +
      `(${cachedSteps}/${totalSteps} from cache); ` +
      `${results.reduce((n, r) => n + r.toolCalls, 0)} tool calls; ` +
      `${results.filter((r) => r.resolution).length}/${TICKETS.length} tickets resolved`,
  );
  return { d, results, totalSteps, cachedSteps };
}

async function main() {
  console.log(`Zerocache: ${BASE_URL}   model: ${MODEL}\n`);
  const runId = Math.random().toString(36).slice(2, 10);

  console.log("Running a 3-ticket triage suite through the agent, twice (a CI/eval loop):");
  const cold = await runSuite(runId, "run 1 (cold)");
  const warm = await runSuite(runId, "run 2 (repeat)");

  console.log();
  check("cold run resolved every ticket", cold.results.every((r) => r.resolution));
  check("cold run actually used tools", cold.results.reduce((n, r) => n + r.toolCalls, 0) >= TICKETS.length);
  check("cold run hit cache zero times", cold.d.hits === 0, `hits=${cold.d.hits} misses=${cold.d.misses}`);
  check("both runs did the same amount of work", warm.totalSteps === cold.totalSteps, `${warm.totalSteps} vs ${cold.totalSteps}`);
  check(
    "repeat run made zero upstream calls",
    warm.d.misses === 0,
    `misses=${warm.d.misses}`,
  );
  check(
    "repeat run: every model call served from cache",
    warm.cachedSteps === warm.totalSteps && warm.d.hits === warm.totalSteps,
    `cached=${warm.cachedSteps}/${warm.totalSteps} hitctr=${warm.d.hits}`,
  );
  check(
    "repeat run saved real input AND output tokens",
    warm.d.promptSaved > 0 && warm.d.completionSaved > 0,
    `${warm.d.promptSaved} prompt + ${warm.d.completionSaved} completion`,
  );
  check(
    "the two runs produced the same resolutions",
    JSON.stringify(cold.results.map((r) => r.resolution)) ===
      JSON.stringify(warm.results.map((r) => r.resolution)),
  );

  const price = PRICE_PER_MTOK[MODEL];
  console.log("\nWhat the repeat run did NOT pay for:");
  console.log(`  prompt tokens:     ${warm.d.promptSaved}`);
  console.log(`  completion tokens: ${warm.d.completionSaved}`);
  if (price) {
    const usd = (warm.d.promptSaved / 1e6) * price.input + (warm.d.completionSaved / 1e6) * price.output;
    console.log(`  ~$${usd.toFixed(6)} per suite run at ${MODEL} list price (illustrative)`);
    console.log(`  a CI pipeline running this suite 200x/day would save ~$${(usd * 200).toFixed(2)}/day,`);
    console.log(`  ~$${(usd * 200 * 30).toFixed(2)}/month -- for a suite that costs nothing extra to keep green.`);
  }

  console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

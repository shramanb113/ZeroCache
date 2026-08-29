// Live smoke test for the OPT-IN semantic completion tier (CLAUDE.md
// Deviations item 24). Proves the thing that tier is sold on: the SAME
// deterministic question, reworded, is served from cache on repeat instead of
// re-billed -- while a materially different question is NOT.
//
// Needs a Zerocache instance built with `--features semantic` and started with
// ZEROCACHE_SEMANTIC=1 on the sled backend, plus a real key. Same
// measure-by-diffing-/metrics technique as battle-test.mjs, no deps.
//
//   ZEROCACHE_SEMANTIC=1 \
//     ZEROCACHE_CHAT_PROVIDERS="gemini=https://generativelanguage.googleapis.com/v1beta/openai" \
//     cargo run -p zerocache-http --features semantic
//   OPENAI_API_KEY=<key> MODEL=gemini-3.5-flash-lite ZEROCACHE_CHAT_PROVIDER=gemini \
//     node demo/completion-cache/paraphrase.mjs
//
// The first request of each group is a real billed call; the rest should hit.

const BASE_URL = (process.env.ZEROCACHE_BASE_URL || "http://localhost:8080").replace(/\/$/, "");
const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.MODEL || "gpt-4o-mini";
const PROVIDER = process.env.ZEROCACHE_CHAT_PROVIDER || "openai";

if (!API_KEY) {
  console.error("OPENAI_API_KEY is required (this test makes real billed calls on the cold call).");
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
    semanticHits: sumOf("zerocache_completion_semantic_hits_total"),
    misses: sumOf("zerocache_completion_cache_misses_total"),
    promptSaved: sumOf("zerocache_completion_prompt_tokens_saved_total"),
    completionSaved: sumOf("zerocache_completion_completion_tokens_saved_total"),
  };
}

// A fresh session id per run keeps the system prompt (and therefore the
// coarse-key hash) unique to this run, so every group starts cold.
const RUN = Math.random().toString(36).slice(2, 8);
const SYSTEM = {
  role: "system",
  content: `You are a support bot. Session ${RUN}. Answer in one short sentence.`,
};

async function chat(userText) {
  const res = await fetch(`${BASE_URL}/${PROVIDER}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({ model: MODEL, messages: [SYSTEM, { role: "user", content: userText }], temperature: 0 }),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`chat -> ${res.status}: ${JSON.stringify(json).slice(0, 300)}`);
  return {
    content: json.choices?.[0]?.message?.content ?? "",
    hit: res.headers.get("x-zerocache-completion-hit") === "true",
    kind: res.headers.get("x-zerocache-completion-hit-kind"),
    score: res.headers.get("x-zerocache-semantic-score"),
  };
}

const BASE_Q = "How do I reset my password?";
const PARAPHRASES = [
  "how can I reset my password",
  "What's the way to reset my password?",
  "i need to reset my password, how?",
  "Reset password — how do I do that?",
  "how do i go about resetting my password",
];
const DIFFERENT_Q = "What are your customer support hours?";

async function main() {
  console.log(`paraphrase smoke -- ${BASE_URL}  provider=${PROVIDER}  model=${MODEL}  run=${RUN}\n`);
  const before = await metricsSnapshot();

  const cold = await chat(BASE_Q);
  check("base question is a cold miss", !cold.hit, `kind=${cold.kind}`);

  let semanticHits = 0;
  for (const p of PARAPHRASES) {
    const r = await chat(p);
    if (r.hit && r.kind === "semantic") {
      semanticHits += 1;
      check(`paraphrase served semantically: "${p}"`, r.content === cold.content, `score=${r.score}, body ${r.content === cold.content ? "identical" : "DIFFERS"}`);
    } else {
      console.log(`  MISS  "${p}" (hit=${r.hit} kind=${r.kind}) -- counts against the 4/5 bar`);
    }
  }
  check("at least 4 of 5 paraphrases are semantic hits", semanticHits >= 4, `${semanticHits}/5`);

  const diff = await chat(DIFFERENT_Q);
  check("a materially different question does NOT hit", !diff.hit, `kind=${diff.kind}`);

  const after = await metricsSnapshot();
  const d = {
    semanticHits: after.semanticHits - before.semanticHits,
    misses: after.misses - before.misses,
    promptSaved: after.promptSaved - before.promptSaved,
    completionSaved: after.completionSaved - before.completionSaved,
  };
  check("semantic-hit counter rose by >= 4", d.semanticHits >= 4, `+${d.semanticHits}`);
  check("miss counter rose by exactly 2 (base + different)", d.misses === 2, `+${d.misses}`);

  console.log(`\ntokens not billed on the paraphrases: ${d.promptSaved} prompt + ${d.completionSaved} completion`);
  console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

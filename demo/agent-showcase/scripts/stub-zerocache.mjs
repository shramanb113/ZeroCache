// Keyless smoke stub: pretends to be a Zerocache instance so `run.ts` can be
// exercised end-to-end with no API keys and no real Zerocache. Not used by the
// real demo. Start it, then:
//   ZEROCACHE_BASE_URL=http://127.0.0.1:8791 OPENAI_API_KEY=stub \
//   ANTHROPIC_API_KEY=stub npx tsx run.ts --run=1
import { createServer } from "node:http";

const PORT = Number(process.env.PORT ?? 8791);
let calls = 0;

const FILE_BODIES = {
  "src/config.ts": `export interface Config {
  port: number;
  rateLimitPerMinute: number;
}
export const config: Config = {
  port: Number(process.env.PORT ?? 3000),
  rateLimitPerMinute: Number(process.env.RATE_LIMIT ?? 60),
};
`,
  "src/rateLimit.ts": `import { config } from "./config.ts";
const buckets = new Map();
export function checkRateLimit(key) {
  const now = Date.now();
  const b = buckets.get(key) ?? { tokens: config.rateLimitPerMinute, ts: now };
  const refill = ((now - b.ts) / 60000) * config.rateLimitPerMinute;
  b.tokens = Math.min(config.rateLimitPerMinute, b.tokens + refill);
  b.ts = now;
  if (b.tokens < 1) {
    buckets.set(key, b);
    return { allowed: false, retryAfter: 1 };
  }
  b.tokens -= 1;
  buckets.set(key, b);
  return { allowed: true, retryAfter: 0 };
}
export function _resetRateLimit() { buckets.clear(); }
`,
  "src/routes/links.ts": `import { addLink, deleteLink, listLinks } from "../store.ts";
import { checkRateLimit } from "../rateLimit.ts";
function json(res, status, body) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}
async function readBody(req) { let b = ""; for await (const c of req) b += c; return b; }
export async function handleLinks(req, res) {
  const url = new URL(req.url ?? "/", "http://localhost");
  const key = req.headers["x-api-key"] ?? "anon";
  const rl = checkRateLimit(key);
  if (!rl.allowed) {
    res.setHeader("Retry-After", String(rl.retryAfter));
    return json(res, 429, { error: "rate limited" });
  }
  if (req.method === "GET" && url.pathname === "/links") return json(res, 200, listLinks());
  if (req.method === "POST" && url.pathname === "/links") {
    const body = await readBody(req);
    let parsed; try { parsed = JSON.parse(body || "{}"); } catch { return json(res, 400, { error: "invalid json" }); }
    if (!parsed.url) return json(res, 400, { error: "url required" });
    return json(res, 201, addLink(parsed.url));
  }
  if (req.method === "DELETE" && url.pathname.startsWith("/links/")) {
    const ok = deleteLink(url.pathname.slice("/links/".length));
    return json(res, ok ? 204 : 404, ok ? {} : { error: "not found" });
  }
  return json(res, 404, { error: "not found" });
}
`,
  "test/rateLimit.test.ts": `import { test } from "node:test";
import assert from "node:assert/strict";
import { checkRateLimit, _resetRateLimit } from "../src/rateLimit.ts";

test("allows the first request and throttles past the cap", () => {
  _resetRateLimit();
  for (let i = 0; i < 60; i++) assert.equal(checkRateLimit("k").allowed, true);
  assert.equal(checkRateLimit("k").allowed, false);
});

test("separate keys have separate buckets", () => {
  _resetRateLimit();
  for (let i = 0; i < 60; i++) checkRateLimit("a");
  assert.equal(checkRateLimit("b").allowed, true);
});
`,
};

function chatResponse(body) {
  const sys = body.messages?.[0]?.content ?? "";
  const user = body.messages?.[1]?.content ?? body.messages?.[0]?.content ?? "";
  if (typeof sys === "string" && sys.includes("unified diff")) {
    const m = /editing exactly one file: (\S+)/.exec(String(user));
    const file = m?.[1];
    return { choices: [{ message: { content: FILE_BODIES[file] ?? "// noop\n" } }], usage: { prompt_tokens: 800, completion_tokens: 200 } };
  }
  if (typeof user === "string" && user.includes("6 bullet points"))
    return { choices: [{ message: { content: "- a\n- b\n- c\n- d\n- e\n- f" } }], usage: { prompt_tokens: 400, completion_tokens: 60 } };
  return { choices: [{ message: { content: "PLAN: create rateLimit.ts, wire routes, add config, add tests" } }], usage: { prompt_tokens: 1200, completion_tokens: 180 } };
}

const server = createServer(async (req, res) => {
  const url = req.url ?? "/";
  if (url === "/health") { res.writeHead(200); return res.end("ok"); }
  if (url === "/metrics") {
    res.writeHead(200, { "content-type": "text/plain" });
    return res.end("zerocache_completion_semantic_hits_total 0\nzerocache_cache_hits_total 0\n");
  }
  let body = "";
  for await (const c of req) body += c;
  const parsed = body ? JSON.parse(body) : {};
  calls++;

  if (url.endsWith("/v1/messages")) {
    res.writeHead(200, { "content-type": "text/event-stream", "x-zerocache-completion-hit": "false" });
    const say = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`);
    say({ type: "message_start", message: { usage: { input_tokens: 900 } } });
    for (const t of ["APPROVED"]) say({ type: "content_block_delta", delta: { type: "text_delta", text: t } });
    say({ type: "message_delta", usage: { output_tokens: 3 } });
    say({ type: "message_stop" });
    return res.end();
  }
  if (url.includes("/v1/embeddings") || url.includes("/v1/images/embeddings")) {
    const n = Array.isArray(parsed.input) ? parsed.input.length : 1;
    res.writeHead(200, { "content-type": "application/json", "x-zerocache-hits": "0", "x-zerocache-misses": String(n) });
    return res.end(JSON.stringify({ object: "list", data: Array.from({ length: n }, (_, i) => ({ embedding: [Math.random(), Math.random(), Math.random()], index: i })), usage: { prompt_tokens: n * 5 } }));
  }
  if (url.includes("/v1/chat/completions")) {
    res.writeHead(200, { "content-type": "application/json", "x-zerocache-completion-hit": "false" });
    return res.end(JSON.stringify(chatResponse(parsed)));
  }
  res.writeHead(404); res.end("nope");
});

server.listen(PORT, () => console.log(`stub-zerocache on :${PORT}`));

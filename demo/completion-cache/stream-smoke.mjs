#!/usr/bin/env node
// Live smoke test for the streaming completion cache (CLAUDE.md Deviations
// item 27). Proves the one thing the feature is sold on: a `stream: true`
// chat completion misses on the first run (streamed live from upstream) and
// replays from cache byte-identically on the second, with
// X-Zerocache-Completion-Hit: true.
//
// Dependency-free -- global fetch + TextDecoder + top-level await, so it runs
// with plain `node` (>= 20), no install. It is the streaming counterpart to
// demo/completion-cache/battle-test.mjs.
//
// Requires a real running Zerocache instance (ZEROCACHE_URL, default
// http://localhost:8080) and a real provider key in API_KEY. It makes one
// real, billed chat call on the first run; the second run bills nothing.
//
//   API_KEY=<key> MODEL=gemini-2.0-flash node demo/completion-cache/stream-smoke.mjs
//
// Env: ZEROCACHE_URL, PROVIDER (default gemini), MODEL (default
// gemini-2.0-flash), API_KEY (required).

const BASE = process.env.ZEROCACHE_URL ?? "http://localhost:8080";
const PROVIDER = process.env.PROVIDER ?? "gemini";
const MODEL = process.env.MODEL ?? "gemini-2.0-flash";
const KEY = process.env.API_KEY;
if (!KEY) { console.error("set API_KEY"); process.exit(2); }

async function once() {
  const res = await fetch(`${BASE}/${PROVIDER}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "Say hello in exactly five words." }],
      temperature: 0,
      stream: true,
    }),
  });
  const hit = res.headers.get("x-zerocache-completion-hit");
  let text = "";
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of res.body) {
    buf += decoder.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      try {
        const j = JSON.parse(payload);
        text += j.choices?.[0]?.delta?.content ?? "";
      } catch {}
    }
  }
  return { hit, text };
}

const a = await once();
const b = await once();
console.log("run 1:", a);
console.log("run 2:", b);
if (a.hit !== "false") { console.error("run 1 should be a miss"); process.exit(1); }
if (b.hit !== "true") { console.error("run 2 should be a hit"); process.exit(1); }
if (a.text !== b.text || !a.text) { console.error("assembled text differs or empty"); process.exit(1); }
console.log("OK: streamed miss then cached SSE replay, byte-identical");

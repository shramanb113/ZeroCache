// Not yet run against a live endpoint (no live Anthropic key in the authoring environment).

import { strict as assert } from "node:assert";

const ZEROCACHE_URL = process.env.ZEROCACHE_URL || "http://localhost:8080";
const API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.MODEL || "claude-opus-4-1-20250805";

if (!API_KEY) {
  console.error("ANTHROPIC_API_KEY env var required");
  process.exit(1);
}

const request = {
  model: MODEL,
  max_tokens: 64,
  temperature: 0,
  messages: [
    {
      role: "user",
      content: "Say hi in one word.",
    },
  ],
};

async function runTest() {
  const headers = {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
  };

  // Run 1: expect cache miss
  const res1 = await fetch(`${ZEROCACHE_URL}/anthropic/v1/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  assert.equal(res1.status, 200, `run 1: expected 200, got ${res1.status}`);
  const hit1 = res1.headers.get("x-zerocache-completion-hit");
  assert.equal(
    hit1,
    "false",
    `run 1: expected x-zerocache-completion-hit: false, got ${hit1}`
  );
  const body1 = await res1.json();
  const text1 = body1.content?.[0]?.text;
  assert(text1, "run 1: response missing content[0].text");

  // Run 2: expect cache hit
  const res2 = await fetch(`${ZEROCACHE_URL}/anthropic/v1/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  assert.equal(res2.status, 200, `run 2: expected 200, got ${res2.status}`);
  const hit2 = res2.headers.get("x-zerocache-completion-hit");
  assert.equal(
    hit2,
    "true",
    `run 2: expected x-zerocache-completion-hit: true, got ${hit2}`
  );
  const body2 = await res2.json();
  const text2 = body2.content?.[0]?.text;
  assert(text2, "run 2: response missing content[0].text");

  // Assert byte-identical
  assert.equal(
    text1,
    text2,
    `text mismatch: run 1 "${text1}" != run 2 "${text2}"`
  );

  console.log("PASS");
}

runTest().catch((err) => {
  console.error("FAIL:", err.message);
  process.exit(1);
});

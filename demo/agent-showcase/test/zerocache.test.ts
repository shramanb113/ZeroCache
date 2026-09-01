import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ZerocacheClient } from "../src/zerocache.ts";
import { Trace } from "../src/trace.ts";

function stub(
  handler: (
    url: string,
    body: string,
  ) => { status?: number; headers?: Record<string, string>; body: string },
) {
  const srv = createServer((req, res) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => {
      const r = handler(req.url ?? "", b);
      res.writeHead(r.status ?? 200, {
        "content-type": "application/json",
        ...r.headers,
      });
      res.end(r.body);
    });
  });
  return new Promise<{ url: string; close: () => void }>((resolve) => {
    srv.listen(0, () => {
      const port = (srv.address() as { port: number }).port;
      resolve({ url: `http://127.0.0.1:${port}`, close: () => srv.close() });
    });
  });
}

test("chat: captures hit headers and writes one trace event", async () => {
  const s = await stub(() => ({
    headers: {
      "x-zerocache-completion-hit": "true",
      "x-zerocache-completion-hit-kind": "exact",
    },
    body: JSON.stringify({
      choices: [{ message: { content: "ok" } }],
      usage: { prompt_tokens: 10, completion_tokens: 2 },
    }),
  }));
  const p = join(mkdtempSync(join(tmpdir(), "zc-")), "run.jsonl");
  const trace = new Trace(p, 2);
  const client = new ZerocacheClient({ baseUrl: s.url, trace });
  const r = await client.chat(
    "openai",
    "sk-test",
    { model: "gpt-4o-mini", messages: [] },
    { stage: "plan" },
  );
  assert.equal(r.hit, true);
  assert.equal(r.hitKind, "exact");
  s.close();
});

test("chat: a miss bills tokens from usage", async () => {
  const s = await stub(() => ({
    headers: { "x-zerocache-completion-hit": "false" },
    body: JSON.stringify({
      choices: [{ message: { content: "ok" } }],
      usage: { prompt_tokens: 100, completion_tokens: 20 },
    }),
  }));
  const p = join(mkdtempSync(join(tmpdir(), "zc-")), "run.jsonl");
  const client = new ZerocacheClient({ baseUrl: s.url, trace: new Trace(p, 1) });
  const r = await client.chat(
    "openai",
    "sk-test",
    { model: "gpt-4o-mini", messages: [] },
    { stage: "plan" },
  );
  assert.equal(r.hit, false);
  s.close();
});

test("embed: hit derived from X-Zerocache-Hits / -Misses headers", async () => {
  const s = await stub(() => ({
    headers: { "x-zerocache-hits": "2", "x-zerocache-misses": "0" },
    body: JSON.stringify({
      object: "list",
      data: [
        { embedding: [1, 0], index: 0 },
        { embedding: [0, 1], index: 1 },
      ],
      usage: { prompt_tokens: 0 },
    }),
  }));
  const p = join(mkdtempSync(join(tmpdir(), "zc-")), "run.jsonl");
  const client = new ZerocacheClient({ baseUrl: s.url, trace: new Trace(p, 2) });
  const r = await client.embed(
    "openai",
    "sk-test",
    "text-embedding-3-small",
    ["a", "b"],
    { stage: "retrieve" },
  );
  assert.equal(r.hit, true);
  assert.deepEqual(r.data, [
    [1, 0],
    [0, 1],
  ]);
  s.close();
});

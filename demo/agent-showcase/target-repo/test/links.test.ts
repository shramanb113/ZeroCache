import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { handleLinks } from "../src/routes/links.ts";
import { _reset } from "../src/store.ts";

function once() {
  return new Promise<{ url: string; close: () => void }>((resolve) => {
    const srv = createServer((req, res) => handleLinks(req, res));
    srv.listen(0, () => {
      const port = (srv.address() as { port: number }).port;
      resolve({ url: `http://127.0.0.1:${port}`, close: () => srv.close() });
    });
  });
}

test("POST then GET /links roundtrips", async () => {
  _reset();
  const s = await once();
  const post = await fetch(`${s.url}/links`, {
    method: "POST",
    body: JSON.stringify({ url: "https://example.com" }),
  });
  assert.equal(post.status, 201);
  const list = (await (await fetch(`${s.url}/links`)).json()) as unknown[];
  assert.equal(list.length, 1);
  s.close();
});

test("POST /links without url is 400", async () => {
  _reset();
  const s = await once();
  const r = await fetch(`${s.url}/links`, { method: "POST", body: "{}" });
  assert.equal(r.status, 400);
  s.close();
});

test("DELETE /links/:id returns 404 for an unknown id", async () => {
  _reset();
  const s = await once();
  const r = await fetch(`${s.url}/links/nope`, { method: "DELETE" });
  assert.equal(r.status, 404);
  s.close();
});

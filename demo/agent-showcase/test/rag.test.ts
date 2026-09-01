import { test } from "node:test";
import assert from "node:assert/strict";
import { cosine, chunkFile, VectorIndex } from "../src/rag.ts";

test("cosine: identical vectors = 1, orthogonal = 0", () => {
  assert.equal(cosine([1, 0, 0], [1, 0, 0]), 1);
  assert.equal(cosine([1, 0, 0], [0, 1, 0]), 0);
});

test("chunkFile: splits on blank lines, stable ids, respects maxChars", () => {
  const content = "para one\n\npara two is here\n\npara three";
  const chunks = chunkFile("src/x.ts", content, 20);
  assert.ok(chunks.length >= 2);
  assert.equal(chunks[0]!.id, "src/x.ts#0");
  assert.ok(chunks.every((c) => c.path === "src/x.ts"));
});

test("VectorIndex.query returns nearest chunk first", () => {
  const idx = new VectorIndex();
  idx.add(
    [
      { id: "a", path: "a", text: "a" },
      { id: "b", path: "b", text: "b" },
    ],
    [
      [1, 0],
      [0, 1],
    ],
  );
  const hits = idx.query([0.9, 0.1], 2);
  assert.equal(hits[0]!.chunk.id, "a");
  assert.ok(hits[0]!.score > hits[1]!.score);
});

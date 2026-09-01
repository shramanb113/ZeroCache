import { test } from "node:test";
import assert from "node:assert/strict";
import { fmtUsd, fmtDur, colorizeDiff, stageCard } from "../src/render.ts";
import { Ledger } from "../src/ledger.ts";

test("fmtUsd / fmtDur", () => {
  assert.equal(fmtUsd(0), "$0.0000");
  assert.equal(fmtUsd(1.5), "$1.50");
  assert.equal(fmtDur(3), "3 ms");
  assert.equal(fmtDur(2100), "2.1 s");
  assert.equal(fmtDur(192000), "3m 12s");
});

test("colorizeDiff keeps every source line", () => {
  const d = "@@ -1 +1 @@\n-old\n+new\n context";
  const out = colorizeDiff(d);
  for (const frag of ["old", "new", "context"]) assert.ok(out.includes(frag));
});

test("stageCard renders the number and name", () => {
  const c = stageCard(3, "BRIEF", "hit", "3 workers -> 1 call");
  assert.ok(c.includes("03"));
  assert.ok(c.includes("BRIEF"));
});

test("Ledger aggregates upstream vs coalesced vs hit", () => {
  const l = new Ledger();
  l.addEvent({
    billedPromptTokens: 100,
    billedCompletionTokens: 10,
    usd: 0.01,
    hit: false,
    coalesced: false,
  });
  l.addEvent({
    billedPromptTokens: 0,
    billedCompletionTokens: 0,
    usd: 0,
    hit: false,
    coalesced: true,
  });
  l.addEvent({
    billedPromptTokens: 0,
    billedCompletionTokens: 0,
    usd: 0,
    hit: true,
    coalesced: false,
  });
  const s = l.snapshot();
  assert.equal(s.upstreamCalls, 1);
  assert.equal(s.promptTokens, 100);
});

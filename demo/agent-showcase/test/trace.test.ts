import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { priceUsd, summarize } from "../src/trace.ts";

test("priceUsd: known model bills in+out, unknown model is free", () => {
  assert.ok(priceUsd("gpt-4o-mini", 1_000_000, 0) > 0);
  assert.equal(priceUsd("no-such-model", 1_000_000, 1_000_000), 0);
});

test("summarize: a hit contributes 0 billed tokens and 0 usd", () => {
  const dir = mkdtempSync(join(tmpdir(), "trace-"));
  const p = join(dir, "run.jsonl");
  const rows = [
    {
      t: 0,
      run: 2,
      stage: "plan",
      type: "chat",
      provider: "openai",
      model: "gpt-4o-mini",
      surface: "chat_completions",
      hit: false,
      hitKind: null,
      semanticScore: null,
      promptTokens: 100,
      completionTokens: 20,
      billedPromptTokens: 100,
      billedCompletionTokens: 20,
      latencyMs: 900,
      usd: 0.001,
      coalesced: false,
      note: "",
    },
    {
      t: 10,
      run: 2,
      stage: "fix",
      type: "chat",
      provider: "openai",
      model: "gpt-4o-mini",
      surface: "chat_completions",
      hit: true,
      hitKind: "exact",
      semanticScore: null,
      promptTokens: 0,
      completionTokens: 0,
      billedPromptTokens: 0,
      billedCompletionTokens: 0,
      latencyMs: 3,
      usd: 0,
      coalesced: false,
      note: "",
    },
  ];
  writeFileSync(p, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const s = summarize(p);
  assert.equal(s.hits, 1);
  assert.equal(s.misses, 1);
  assert.equal(s.upstreamCalls, 1);
  assert.equal(s.billedPromptTokens, 100);
});

test("summarize: coalesced calls are not counted as upstream calls", () => {
  const dir = mkdtempSync(join(tmpdir(), "trace-"));
  const p = join(dir, "run.jsonl");
  const base = {
    t: 0,
    run: 1 as const,
    stage: "brief",
    type: "chat" as const,
    provider: "openai",
    model: "gpt-4o-mini",
    surface: "chat_completions" as const,
    hit: false,
    hitKind: null,
    semanticScore: null,
    promptTokens: 50,
    completionTokens: 10,
    latencyMs: 800,
    usd: 0.0005,
    note: "",
  };
  const rows = [
    { ...base, billedPromptTokens: 50, billedCompletionTokens: 10, coalesced: false },
    { ...base, billedPromptTokens: 0, billedCompletionTokens: 0, coalesced: true },
    { ...base, billedPromptTokens: 0, billedCompletionTokens: 0, coalesced: true },
  ];
  writeFileSync(p, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const s = summarize(p);
  assert.equal(s.upstreamCalls, 1);
  assert.equal(s.coalescedCalls, 2);
});

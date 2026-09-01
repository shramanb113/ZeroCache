import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, cpSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { orchestrate } from "../src/orchestrator.ts";
import { RATE_LIMIT_TASK } from "../src/agents.ts";
import { Trace } from "../src/trace.ts";
import { Ledger } from "../src/ledger.ts";

function fakeGateway() {
  const calls: Array<{ kind: string; stage: string; body?: unknown; n?: number }> =
    [];
  const ok = {
    hit: false,
    hitKind: null,
    semanticScore: null,
    hitsHeader: null,
    missesHeader: null,
    latencyMs: 1,
    billedPromptTokens: 5,
    billedCompletionTokens: 5,
    usd: 0,
    coalesced: false,
  };
  return {
    calls,
    async chat(_p: string, _k: string, body: any, meta: any) {
      calls.push({ kind: "chat", stage: meta.stage, body });
      const content =
        meta.stage === "plan"
          ? "PLAN: create src/rateLimit.ts, edit routes/config, add tests"
          : meta.stage === "brief"
            ? "- node:http\n- ESM\n- node:test\n- Map store\n- JSON helpers\n- URL routing"
            : "--- a/src/x.ts\n+++ b/src/x.ts\n@@ -1 +1 @@\n-a\n+b";
      return {
        ...ok,
        data: {
          choices: [{ message: { content } }],
          usage: { prompt_tokens: 5, completion_tokens: 5 },
        },
      };
    },
    async messagesStream(
      _p: string,
      _k: string,
      _b: any,
      onDelta: (t: string) => void,
      meta: any,
    ) {
      calls.push({ kind: "messages", stage: meta.stage });
      onDelta("APPROVED");
      return { ...ok, data: { text: "APPROVED", raw: {} } };
    },
    async embed(_p: string, _k: string, _m: string, input: string[], meta: any) {
      calls.push({ kind: "embed", stage: meta.stage, n: input.length });
      return { ...ok, data: input.map(() => [1, 0, 0]) };
    },
    async embedImages(
      _p: string,
      _k: string,
      _m: string,
      uris: string[],
      meta: any,
    ) {
      calls.push({ kind: "embedImages", stage: meta.stage, n: uris.length });
      return { ...ok, data: uris.map(() => [0, 1, 0]) };
    },
  };
}

test("orchestrate runs every stage and fires 3 concurrent brief calls", async () => {
  const work = mkdtempSync(join(tmpdir(), "work-"));
  cpSync(join(import.meta.dirname, "../target-repo"), work, { recursive: true });
  const gw = fakeGateway();
  const p = join(mkdtempSync(join(tmpdir(), "t-")), "run.jsonl");
  const res = await orchestrate(RATE_LIMIT_TASK, {
    gateway: gw as never,
    keys: { openai: "sk-x", anthropic: "sk-ant-x" },
    providers: { chat: "openai", embed: "openai" },
    models: {
      chat: "gpt-4o-mini",
      review: "claude-sonnet-5",
      embed: "text-embedding-3-small",
      image: "gemini-embedding-001",
    },
    workDir: work,
    trace: new Trace(p, 1),
    ledger: new Ledger(),
    onFrame: () => {},
    run: 1,
  });
  const briefCalls = gw.calls.filter((c) => c.stage === "brief");
  assert.equal(briefCalls.length, 3);
  assert.ok(gw.calls.some((c) => c.stage === "plan"));
  assert.ok(gw.calls.some((c) => c.kind === "messages"));
  assert.ok(res.stages.length >= 6);
});

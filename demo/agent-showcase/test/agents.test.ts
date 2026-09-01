import { test } from "node:test";
import assert from "node:assert/strict";
import {
  RATE_LIMIT_TASK,
  RATE_LIMIT_TASK_PARAPHRASED,
  architectPlanBody,
  repoBriefBody,
  coderBody,
  reviewerMessagesBody,
  fixerBody,
} from "../src/agents.ts";

test("every chat body pins temperature 0 and carries >=2 messages", () => {
  const b = architectPlanBody("gpt-4o-mini", RATE_LIMIT_TASK, []) as any;
  assert.equal(b.temperature, 0);
  assert.equal(b.model, "gpt-4o-mini");
  assert.ok(Array.isArray(b.messages) && b.messages.length >= 2);

  const c = coderBody(
    "gpt-4o-mini",
    RATE_LIMIT_TASK,
    "src/rateLimit.ts",
    "",
    "PLAN",
  ) as any;
  assert.equal(c.temperature, 0);
  assert.equal(c.stream, undefined);

  const f = fixerBody(
    "gpt-4o-mini",
    RATE_LIMIT_TASK,
    "src/rateLimit.ts",
    "x",
    "REVIEW",
  ) as any;
  assert.equal(f.temperature, 0);
});

test("repoBriefBody is byte-identical for identical conventions text (coalescing precondition)", () => {
  const a = JSON.stringify(repoBriefBody("gpt-4o-mini", "conv"));
  const b = JSON.stringify(repoBriefBody("gpt-4o-mini", "conv"));
  assert.equal(a, b);
});

test("reviewer body is Anthropic-shaped, temperature 0, streaming", () => {
  const b = reviewerMessagesBody("claude-sonnet-5", RATE_LIMIT_TASK, [
    { path: "src/rateLimit.ts", unifiedDiff: "@@ -0,0 +1 @@\n+x" },
  ]) as any;
  assert.equal(b.temperature, 0);
  assert.equal(b.stream, true);
  assert.ok(typeof b.max_tokens === "number");
  assert.equal(b.messages[0].role, "user");
});

test("paraphrased task keeps title and targets, changes brief", () => {
  assert.equal(RATE_LIMIT_TASK_PARAPHRASED.title, RATE_LIMIT_TASK.title);
  assert.deepEqual(
    RATE_LIMIT_TASK_PARAPHRASED.targetFiles,
    RATE_LIMIT_TASK.targetFiles,
  );
  assert.notEqual(RATE_LIMIT_TASK_PARAPHRASED.brief, RATE_LIMIT_TASK.brief);
});

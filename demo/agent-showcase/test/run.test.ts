import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

test("--check exits 0 with a skip message when OPENAI_API_KEY is unset", () => {
  const out = execFileSync(
    process.execPath,
    ["--import", "tsx", "run.ts", "--run=2", "--check"],
    {
      cwd: join(import.meta.dirname, ".."),
      env: { ...process.env, OPENAI_API_KEY: "" },
      encoding: "utf8",
    },
  );
  assert.match(out, /skipped \(no OPENAI_API_KEY\)/);
});

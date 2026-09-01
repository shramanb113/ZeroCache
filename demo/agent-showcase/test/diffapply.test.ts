import { test } from "node:test";
import assert from "node:assert/strict";
import { applyUnifiedDiff, DiffApplyError } from "../src/diffapply.ts";

const ORIGINAL = `line one
line two
line three
`;

test("applies a clean single-hunk diff", () => {
  const d = `--- a/x.ts
+++ b/x.ts
@@ -1,3 +1,3 @@
 line one
-line two
+line two changed
 line three
`;
  assert.equal(
    applyUnifiedDiff(ORIGINAL, d),
    "line one\nline two changed\nline three\n",
  );
});

test("applies a pure-addition hunk", () => {
  const d = `--- a/x.ts
+++ b/x.ts
@@ -3,1 +3,2 @@
 line three
+line four
`;
  assert.equal(
    applyUnifiedDiff(ORIGINAL, d),
    "line one\nline two\nline three\nline four\n",
  );
});

test("throws DiffApplyError when context does not match", () => {
  const d = `--- a/x.ts
+++ b/x.ts
@@ -1,3 +1,3 @@
 line ONE
-line two
+nope
 line three
`;
  assert.throws(() => applyUnifiedDiff(ORIGINAL, d), DiffApplyError);
});

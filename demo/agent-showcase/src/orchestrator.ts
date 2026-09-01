import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname, relative, sep } from "node:path";
import type { Trace } from "./trace.ts";
import type { Ledger } from "./ledger.ts";
import type { ZcResult } from "./zerocache.ts";
import { VectorIndex, chunkFile, type Chunk } from "./rag.ts";
import { applyUnifiedDiff, DiffApplyError } from "./diffapply.ts";
import { runNodeTests } from "./verify.ts";
import {
  architectPlanBody,
  repoBriefBody,
  coderBody,
  reviewerMessagesBody,
  fixerBody,
  type Task,
} from "./agents.ts";

export interface LlmGateway {
  chat(
    provider: string,
    apiKey: string,
    body: object,
    meta: { stage: string; coalesced?: boolean },
  ): Promise<ZcResult<Record<string, unknown>>>;
  messagesStream(
    provider: string,
    apiKey: string,
    body: object,
    onDelta: (text: string) => void,
    meta: { stage: string },
  ): Promise<ZcResult<{ text: string; raw: unknown }>>;
  embed(
    provider: string,
    apiKey: string,
    model: string,
    input: string[],
    meta: { stage: string },
  ): Promise<ZcResult<number[][]>>;
  embedImages(
    provider: string,
    apiKey: string,
    model: string,
    dataUris: string[],
    meta: { stage: string },
  ): Promise<ZcResult<number[][]>>;
}

export interface OrchestratorDeps {
  gateway: LlmGateway;
  keys: { openai: string; anthropic?: string; gemini?: string };
  /** Which registered Zerocache providers to route through. Default openai. */
  providers: { chat: string; embed: string };
  models: { chat: string; review: string; embed: string; image: string };
  workDir: string;
  trace: Trace;
  ledger: Ledger;
  onFrame: (stages: StageOutcome[], streamText?: string) => void;
  run: 1 | 2 | 3;
}

export interface StageOutcome {
  name: string;
  status: "done" | "hit" | "failed";
  detail: string;
}

export interface OrchestratorResult {
  testPass: boolean;
  testsPassed: number;
  testsFailed: number;
  stages: StageOutcome[];
  workDir: string;
  reviewText: string;
  degraded: string[];
}

const SOURCE_EXT = [".ts", ".md"];

function walkSources(root: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(root, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const dir = (entry as unknown as { parentPath?: string; path?: string }).parentPath ??
      (entry as unknown as { path: string }).path;
    const full = join(dir, entry.name);
    const rel = relative(root, full).split(sep).join("/");
    if (rel.startsWith("node_modules/")) continue;
    if (!SOURCE_EXT.some((e) => rel.endsWith(e))) continue;
    out.push(rel);
  }
  return out.sort();
}

function stripFences(s: string): string {
  const m = /^```(?:diff|ts|typescript)?\n([\s\S]*?)\n```$/m.exec(s.trim());
  return (m?.[1] ?? s).trim();
}

function looksLikeDiff(s: string): boolean {
  return s.includes("@@") || s.startsWith("--- ") || s.startsWith("diff --git");
}

/** Apply a coder/fixer response to a file's current content, tolerantly. */
function applyCoderOutput(current: string, raw: string): string {
  const cleaned = stripFences(raw);
  if (looksLikeDiff(cleaned)) {
    try {
      return applyUnifiedDiff(current, cleaned);
    } catch (e) {
      if (current === "") {
        const added = cleaned
          .split("\n")
          .filter((l) => l.startsWith("+") && !l.startsWith("+++"))
          .map((l) => l.slice(1))
          .join("\n");
        if (added.trim().length > 0) return added + "\n";
      }
      throw e;
    }
  }
  // The model returned a full file body instead of a diff.
  if (/\b(export|import|function|const|class)\b/.test(cleaned)) {
    return cleaned.endsWith("\n") ? cleaned : cleaned + "\n";
  }
  throw new DiffApplyError("response was neither a diff nor a file body", "");
}

function readFileOrEmpty(root: string, rel: string): string {
  const full = join(root, rel);
  return existsSync(full) ? readFileSync(full, "utf8") : "";
}

/**
 * Rewrite relative imports the model got the depth wrong on. For a file at
 * `rel`, any `from "./x.ts"` / `"../x.ts"` whose target does not exist is
 * re-pointed at the unique file named `x.ts` among `knownFiles`, path made
 * relative to `rel`'s directory. Deterministic, and only ever narrows a broken
 * import to an existing one.
 */
function repairImports(rel: string, content: string, knownFiles: string[]): string {
  const fromDir = dirname(rel);
  return content.replace(
    /(\bfrom\s+["'])(\.\.?\/[^"']+?\.ts)(["'])/g,
    (whole, pre: string, spec: string, post: string) => {
      const resolved = join(fromDir, spec).split(sep).join("/");
      if (knownFiles.includes(resolved)) return whole;
      const base = spec.split("/").pop();
      const matches = knownFiles.filter((f) => f.split("/").pop() === base);
      if (matches.length !== 1) return whole;
      let repl = relative(fromDir, matches[0]!).split(sep).join("/");
      if (!repl.startsWith(".")) repl = `./${repl}`;
      return `${pre}${repl}${post}`;
    },
  );
}

function writeFileRel(root: string, rel: string, content: string): void {
  const full = join(root, rel);
  mkdirSync(dirname(full), { recursive: true });
  writeFileSync(full, content);
}

export async function orchestrate(
  task: Task,
  deps: OrchestratorDeps,
): Promise<OrchestratorResult> {
  const { gateway, keys, providers, models, workDir, ledger, onFrame } = deps;
  const stages: StageOutcome[] = [];
  const degraded: string[] = [];
  const record = (r: ZcResult<unknown>) =>
    ledger.addEvent({
      billedPromptTokens: r.billedPromptTokens,
      billedCompletionTokens: r.billedCompletionTokens,
      usd: r.usd,
      hit: r.hit,
      coalesced: r.coalesced,
    });

  // ---- Stage 1: retrieve ------------------------------------------------
  const files = walkSources(workDir);
  const chunks: Chunk[] = [];
  for (const rel of files) {
    const text = readFileSync(join(workDir, rel), "utf8");
    chunks.push(...chunkFile(rel, text));
  }
  const embedRes = await gateway.embed(
    providers.embed,
    keys.openai,
    models.embed,
    chunks.map((c) => c.text),
    { stage: "retrieve" },
  );
  record(embedRes);
  const index = new VectorIndex();
  index.add(chunks, embedRes.data);

  let sawImage = false;
  if (keys.gemini && existsSync(join(workDir, "docs/architecture.png"))) {
    try {
      const b64 = readFileSync(join(workDir, "docs/architecture.png")).toString(
        "base64",
      );
      const imgRes = await gateway.embedImages(
        "gemini",
        keys.gemini,
        models.image,
        [`data:image/png;base64,${b64}`],
        { stage: "retrieve" },
      );
      record(imgRes);
      sawImage = true;
    } catch {
      degraded.push("image step failed");
    }
  }
  // No GEMINI_API_KEY is the default: the image-embeddings surface is optional
  // multimodal breadth, not part of the hero path, so its absence is not a
  // degradation.

  const queryRes = await gateway.embed(
    providers.embed,
    keys.openai,
    models.embed,
    [`${task.title}\n${task.brief}`],
    { stage: "retrieve" },
  );
  record(queryRes);
  const top = index.query(queryRes.data[0] ?? [], 8);
  const allHit = embedRes.hit && queryRes.hit;
  stages.push({
    name: "RETRIEVE",
    status: allHit ? "hit" : "done",
    detail: `${index.size} chunks · ${sawImage ? "arch.png" : "no image"}`,
  });
  onFrame(stages);

  // ---- Stage 2: plan --------------------------------------------------
  const planRes = await gateway.chat(
    providers.chat,
    keys.openai,
    architectPlanBody(models.chat, task, top.map((t) => t.chunk)),
    { stage: "plan" },
  );
  record(planRes);
  const plan =
    ((planRes.data.choices as { message: { content: string } }[] | undefined)?.[0]
      ?.message.content) ?? "(no plan)";
  stages.push({
    name: "PLAN",
    status: planRes.hit ? "hit" : "done",
    detail: planRes.hit ? "exact · 0 ms billed" : `${planRes.latencyMs} ms`,
  });
  onFrame(stages);

  // ---- Stage 3: brief (3 concurrent identical calls -> coalesced) -----
  const conventionsText = top
    .slice(0, 3)
    .map((t) => `// ${t.chunk.path}\n${t.chunk.text}`)
    .join("\n\n");
  const briefResults = await Promise.all(
    [0, 1, 2].map((i) =>
      gateway.chat(
        providers.chat,
        keys.openai,
        repoBriefBody(models.chat, conventionsText),
        { stage: "brief", coalesced: i !== 0 },
      ),
    ),
  );
  briefResults.forEach(record);
  const briefHits = briefResults.filter((r) => r.hit).length;
  stages.push({
    name: "BRIEF",
    status: briefHits === 3 ? "hit" : "done",
    detail:
      briefHits === 3
        ? "3 workers · all cached"
        : "3 workers → 1 upstream call (coalesced)",
  });
  onFrame(stages);

  // ---- Stage 4: implement (parallel workers, one file each) ----------
  const knownFiles = [...new Set([...files, ...task.targetFiles])].sort();
  let implemented = 0;
  const failedFiles: string[] = [];
  // Sequential: each worker sees what the previous ones wrote, so a small model
  // keeps cross-file names consistent instead of collapsing every file to one
  // answer. (The BRIEF stage is where concurrent coalescing is demonstrated.)
  for (let w = 0; w < task.targetFiles.length; w++) {
    const rel = task.targetFiles[w]!;
    const liveSnapshot = task.targetFiles.map((r) => ({
      path: r,
      content: readFileOrEmpty(workDir, r),
    }));
    const current = readFileOrEmpty(workDir, rel);
    let attempt = 0;
    let lastErr = "";
    let done = false;
    while (attempt < 2 && !done) {
      attempt++;
      const body = coderBody(
        models.chat,
        task,
        rel,
        current,
        attempt === 1 ? plan : `${plan}\n\nPREVIOUS ATTEMPT FAILED: ${lastErr}`,
        liveSnapshot,
      );
      let res;
      try {
        res = await gateway.chat(providers.chat, keys.openai, body, {
          stage: "implement",
        });
      } catch (e) {
        lastErr = (e as Error).message;
        continue;
      }
      record(res);
      const diff =
        ((res.data.choices as { message: { content: string } }[] | undefined)?.[0]
          ?.message.content) ?? "";
      try {
        const body2 = repairImports(
          rel,
          applyCoderOutput(current, diff),
          knownFiles,
        );
        writeFileRel(workDir, rel, body2);
        implemented++;
        done = true;
      } catch (e) {
        lastErr = (e as Error).message;
      }
    }
    if (!done) failedFiles.push(rel);
    onFrame(stages);
  }
  stages.push({
    name: "IMPLEMENT",
    status: failedFiles.length === 0 ? "done" : "failed",
    detail:
      failedFiles.length === 0
        ? `${implemented}/${task.targetFiles.length} files`
        : `failed: ${failedFiles.join(", ")}`,
  });
  onFrame(stages);

  // ---- Stage 5: review (Claude via /v1/messages, streaming) ---------
  const currentDiffs = task.targetFiles.map((rel) => ({
    path: rel,
    unifiedDiff: readFileOrEmpty(workDir, rel).slice(0, 4000),
  }));
  let reviewText = "";
  if (keys.anthropic) {
    const rev = await gateway.messagesStream(
      "anthropic",
      keys.anthropic,
      reviewerMessagesBody(models.review, task, currentDiffs),
      (t) => {
        reviewText += t;
        onFrame(stages, t);
      },
      { stage: "review" },
    );
    record(rev);
    reviewText = rev.data.text || reviewText;
    stages.push({
      name: "REVIEW",
      status: rev.hit ? "hit" : "done",
      detail: rev.hit ? "replayed from cache" : `${rev.latencyMs} ms streamed`,
    });
  } else {
    degraded.push("anthropic→openai");
    const reviewPrompt =
      (
        reviewerMessagesBody(models.chat, task, currentDiffs) as {
          messages: { content: string }[];
        }
      ).messages[0]?.content ?? "Review this change.";
    const rev = await gateway.chat(
      providers.chat,
      keys.openai,
      {
        model: models.chat,
        temperature: 0,
        messages: [{ role: "user", content: reviewPrompt }],
      },
      { stage: "review" },
    );
    record(rev);
    reviewText =
      ((rev.data.choices as { message: { content: string } }[] | undefined)?.[0]
        ?.message.content) ?? "APPROVED";
    deps.trace.add({
      stage: "review",
      type: "note",
      provider: providers.chat,
      model: models.chat,
      surface: null,
      hit: false,
      hitKind: null,
      semanticScore: null,
      promptTokens: 0,
      completionTokens: 0,
      billedPromptTokens: 0,
      billedCompletionTokens: 0,
      latencyMs: 0,
      usd: 0,
      coalesced: false,
      note: "degraded: anthropic->openai",
    });
    stages.push({
      name: "REVIEW",
      status: rev.hit ? "hit" : "done",
      detail: "openai fallback",
    });
  }
  onFrame(stages);

  // ---- Stage 6: fix -------------------------------------------------
  const approved =
    /^\s*APPROVED\s*$/im.test(reviewText.trim()) && failedFiles.length === 0;
  if (!approved) {
    const fixSnapshot = task.targetFiles.map((rel) => ({
      path: rel,
      content: readFileOrEmpty(workDir, rel),
    }));
    for (const rel of task.targetFiles) {
      const current = readFileOrEmpty(workDir, rel);
      const res = await gateway.chat(
        providers.chat,
        keys.openai,
        fixerBody(models.chat, task, rel, current, reviewText, fixSnapshot),
        { stage: "fix" },
      );
      record(res);
      const diff =
        ((res.data.choices as { message: { content: string } }[] | undefined)?.[0]
          ?.message.content) ?? "";
      try {
        const next = repairImports(
          rel,
          applyCoderOutput(current, diff),
          knownFiles,
        );
        if (next.trim().length > 0) writeFileRel(workDir, rel, next);
      } catch {
        /* keep the pre-fix version */
      }
    }
    stages.push({ name: "FIX", status: "done", detail: "review applied" });
  } else {
    stages.push({
      name: "FIX",
      status: "done",
      detail: approved ? "approved, no changes" : "skipped",
    });
  }
  onFrame(stages);

  // ---- Stage 7: verify -------------------------------------------
  const tests = await runNodeTests(workDir);
  stages.push({
    name: "VERIFY",
    status: tests.ok ? "done" : "failed",
    detail: `${tests.passed} passed, ${tests.failed} failed`,
  });
  deps.trace.add({
    stage: "verify",
    type: "test",
    provider: "",
    model: "",
    surface: null,
    hit: false,
    hitKind: null,
    semanticScore: null,
    promptTokens: 0,
    completionTokens: 0,
    billedPromptTokens: 0,
    billedCompletionTokens: 0,
    latencyMs: 0,
    usd: 0,
    coalesced: false,
    note: `tests ${tests.passed}/${tests.passed + tests.failed}`,
  });
  onFrame(stages);

  return {
    testPass: tests.ok,
    testsPassed: tests.passed,
    testsFailed: tests.failed,
    stages,
    workDir,
    reviewText,
    degraded,
  };
}

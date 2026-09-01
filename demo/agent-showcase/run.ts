import { config as loadEnv } from "dotenv";
import {
  rmSync,
  cpSync,
  mkdirSync,
  existsSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { join, relative, sep } from "node:path";
import { createHash } from "node:crypto";
import { Trace, summarize, compare } from "./src/trace.ts";
import { Ledger } from "./src/ledger.ts";
import { ZerocacheClient } from "./src/zerocache.ts";
import { orchestrate, type OrchestratorResult } from "./src/orchestrator.ts";
import { RATE_LIMIT_TASK, RATE_LIMIT_TASK_PARAPHRASED } from "./src/agents.ts";
import { Board } from "./src/board.ts";
import { savingsReport } from "./src/render.ts";

loadEnv();

const HERE = import.meta.dirname;
const TARGET = join(HERE, "target-repo");
const TRACES = join(HERE, "traces");
const BASE = process.env.ZEROCACHE_BASE_URL ?? "http://localhost:8080";

const RUN_NAME: Record<1 | 2 | 3, string> = {
  1: "cold",
  2: "warm",
  3: "semantic",
};

interface Flags {
  run: 1 | 2 | 3;
  check: boolean;
  record: boolean;
}

function parseFlags(argv: string[]): Flags {
  const runArg = argv.find((a) => a.startsWith("--run="));
  const run = (runArg ? Number(runArg.slice(6)) : 1) as 1 | 2 | 3;
  return {
    run: run === 2 || run === 3 ? run : 1,
    check: argv.includes("--check"),
    record: argv.includes("--record"),
  };
}

async function preflightHealth(): Promise<void> {
  try {
    const r = await fetch(`${BASE}/health`);
    if (r.ok) return;
  } catch {
    /* fallthrough */
  }
  console.error(
    `\nZerocache is not answering at ${BASE}/health.\n` +
      `Start it first:  cargo run -p zerocache-http` +
      `  (add --features semantic for --run=3)\n`,
  );
  process.exit(1);
}

async function semanticEnabled(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/metrics`);
    const body = await r.text();
    return body.includes("zerocache_completion_semantic_hits_total");
  } catch {
    return false;
  }
}

function hashWorkDir(dir: string): string {
  const rels: string[] = [];
  for (const e of readdirSync(dir, { recursive: true, withFileTypes: true })) {
    if (!e.isFile()) continue;
    const d =
      (e as unknown as { parentPath?: string }).parentPath ??
      (e as unknown as { path: string }).path;
    rels.push(relative(dir, join(d, e.name)).split(sep).join("/"));
  }
  rels.sort();
  const h = createHash("sha256");
  for (const rel of rels) {
    h.update(rel);
    h.update("\0");
    h.update(readFileSync(join(dir, rel)));
    h.update("\0");
  }
  return h.digest("hex");
}

export async function executeRun(
  flags: Flags,
  opts: { quiet?: boolean } = {},
): Promise<{ result: OrchestratorResult; tracePath: string }> {
  const workDir = join(HERE, ".work");
  rmSync(workDir, { recursive: true, force: true });
  cpSync(TARGET, workDir, { recursive: true });

  mkdirSync(TRACES, { recursive: true });
  const tracePath = join(TRACES, `run-${RUN_NAME[flags.run]}.jsonl`);
  const trace = new Trace(tracePath, flags.run);
  const ledger = new Ledger();

  const keys = {
    openai: process.env.OPENAI_API_KEY ?? "",
    anthropic: process.env.ANTHROPIC_API_KEY || undefined,
    gemini: process.env.GEMINI_API_KEY || undefined,
  };
  const models = {
    chat: process.env.SHOWCASE_CHAT_MODEL ?? "gpt-4o-mini",
    review: process.env.SHOWCASE_REVIEW_MODEL ?? "claude-sonnet-5",
    embed: process.env.SHOWCASE_EMBED_MODEL ?? "text-embedding-3-small",
    image: process.env.SHOWCASE_IMAGE_MODEL ?? "gemini-embedding-001",
  };

  const client = new ZerocacheClient({ baseUrl: BASE, trace });
  const task = flags.run === 3 ? RATE_LIMIT_TASK_PARAPHRASED : RATE_LIMIT_TASK;

  const providers = ["openai", keys.anthropic ? "anthropic" : null, keys.gemini ? "gemini" : null].filter(
    Boolean,
  ) as string[];
  const coldTrace = join(TRACES, "run-cold.jsonl");
  const run1View =
    flags.run !== 1 && existsSync(coldTrace) ? summarize(coldTrace) : undefined;

  const board = opts.quiet
    ? undefined
    : new Board(
          {
            task: task.title,
            run: flags.run,
            mode: RUN_NAME[flags.run],
            providers,
            degraded: [],
          },
          ledger,
          run1View,
        );

  let streamBuf = "";
  const result = await orchestrate(task, {
    gateway: client,
    keys,
    models,
    workDir,
    trace,
    ledger,
    run: flags.run,
    onFrame: (stages, streamText) => {
      if (!board) return;
      if (streamText) {
        streamBuf += streamText;
        board.setStream(streamBuf);
      }
      board.draw(stages);
    },
  });
  trace.close();
  if (board) board.draw(result.stages);

  return { result, tracePath };
}

async function main(): Promise<void> {
  const flags = parseFlags(process.argv.slice(2));

  if (!process.env.OPENAI_API_KEY) {
    if (flags.check) {
      console.log("check skipped (no OPENAI_API_KEY)");
      process.exit(0);
    }
    console.error("OPENAI_API_KEY is required (see .env.example).");
    process.exit(1);
  }

  await preflightHealth();

  if (flags.run === 3 && !(await semanticEnabled())) {
    console.log(
      "\nsemantic tier not enabled on this Zerocache instance.\n" +
        "Start it with:  cargo run -p zerocache-http --features semantic\n" +
        "and set        ZEROCACHE_SEMANTIC=1\n",
    );
    process.exit(0);
  }

  const runs: (1 | 2 | 3)[] = flags.record ? [1, 2, 3] : [flags.run];
  let lastResult: OrchestratorResult | undefined;
  for (const run of runs) {
    if (run === 3 && !(await semanticEnabled())) {
      console.log("\n(skipping run 3 — semantic tier not enabled)\n");
      break;
    }
    const { result, tracePath } = await executeRun({ ...flags, run });
    lastResult = result;

    if (run === 1) {
      writeFileSync(
        join(TRACES, ".work-run1-hash.json"),
        JSON.stringify({ sha256: hashWorkDir(join(HERE, ".work")) }, null, 2),
      );
    }

    console.log();
    console.log(
      `run ${run} (${RUN_NAME[run]}): ${result.testsPassed} tests passed, ` +
        `${result.testsFailed} failed`,
    );

    if (flags.check && run === 2) {
      const s = summarize(tracePath);
      const stored = existsSync(join(TRACES, ".work-run1-hash.json"))
        ? JSON.parse(readFileSync(join(TRACES, ".work-run1-hash.json"), "utf8"))
            .sha256
        : null;
      const nowHash = hashWorkDir(join(HERE, ".work"));
      const problems: string[] = [];
      if (s.misses > 0) problems.push(`${s.misses} cache misses on the warm run`);
      if (stored && stored !== nowHash)
        problems.push("warm .work/ differs from run 1");
      if (problems.length > 0) {
        console.error("CHECK FAILED: " + problems.join("; "));
        process.exit(1);
      }
      console.log("CHECK PASSED: warm run was all hits and byte-identical.");
    }
  }

  const coldP = join(TRACES, "run-cold.jsonl");
  const warmP = join(TRACES, "run-warm.jsonl");
  const semP = join(TRACES, "run-semantic.jsonl");
  if (existsSync(coldP) && existsSync(warmP)) {
    console.log();
    console.log(
      savingsReport(
        compare(coldP, warmP, existsSync(semP) ? semP : warmP),
      ),
    );
  }

  if (lastResult && !lastResult.testPass && !flags.record) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

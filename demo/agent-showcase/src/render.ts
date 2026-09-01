import type { RunComparison } from "./trace.ts";

const COLOR =
  !process.env.NO_COLOR &&
  !process.argv.includes("--no-color") &&
  Boolean(process.stdout.isTTY);

function c(open: string, s: string): string {
  return COLOR ? `\x1b[${open}m${s}\x1b[0m` : s;
}
const rust = (s: string) => c("38;2;196;98;47", s);
const green = (s: string) => c("38;2;63;163;114", s);
const amber = (s: string) => c("38;2;184;132;46", s);
const muted = (s: string) => c("38;2;138;133;120", s);
const bold = (s: string) => c("1", s);
const dim = (s: string) => c("2", s);

export function fmtUsd(n: number): string {
  return "$" + n.toFixed(n < 1 ? 4 : 2);
}

export function fmtDur(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function colorizeDiff(unifiedDiff: string): string {
  return unifiedDiff
    .split("\n")
    .map((line) => {
      if (line.startsWith("@@")) return rust(line);
      if (line.startsWith("+++") || line.startsWith("---")) return dim(line);
      if (line.startsWith("+")) return green(line);
      if (line.startsWith("-")) return amber(line);
      return line;
    })
    .join("\n");
}

type StageStatus = "pending" | "running" | "done" | "hit";

const GLYPH: Record<StageStatus, string> = {
  pending: "·",
  running: "●",
  done: "✓",
  hit: "⚡",
};

export function stageCard(
  n: number,
  name: string,
  status: StageStatus,
  detail: string,
): string {
  const num = String(n).padStart(2, "0");
  const g =
    status === "done" || status === "hit"
      ? green(GLYPH[status])
      : status === "running"
        ? rust(GLYPH[status])
        : muted(GLYPH[status]);
  const label = status === "pending" ? muted(name) : bold(name);
  return `  ${rust("§" + num)}  ${label.padEnd(20)} ${g}  ${muted(detail)}`;
}

export interface HeaderOpts {
  task: string;
  run: 1 | 2 | 3;
  mode: string;
  providers: string[];
  degraded: string[];
}

export function header(opts: HeaderOpts): string {
  const line = "─".repeat(64);
  const deg =
    opts.degraded.length > 0
      ? `\n  ${amber("degraded: " + opts.degraded.join(" · "))}`
      : "";
  return [
    rust(`┌${line}┐`),
    `  ${bold("ZEROCACHE")} ${muted("· agent showcase")}`,
    `  ${muted("task:")} ${opts.task}`,
    `  ${muted("run")} ${opts.run} ${muted("of 3 ·")} ${opts.mode} ${muted("· providers:")} ${opts.providers.join(" · ")}${deg}`,
    rust(`└${line}┘`),
  ].join("\n");
}

interface LedgerView {
  upstreamCalls: number;
  promptTokens: number;
  completionTokens: number;
  usd: number;
  wallMs: number;
}

function pad(s: string, w: number): string {
  return s.length >= w ? s : s + " ".repeat(w - s.length);
}

export function ledgerPanel(now: LedgerView, run1?: LedgerView): string {
  const line = "─".repeat(60);
  const ref = (v: string) => (run1 ? muted(`  (run 1: ${v})`) : "");
  const rows = [
    `upstream calls   ${pad(String(now.upstreamCalls), 8)}${ref(String(run1?.upstreamCalls ?? ""))}`,
    `prompt tokens    ${pad(now.promptTokens.toLocaleString(), 8)}${ref((run1?.promptTokens ?? 0).toLocaleString())}`,
    `completion tok   ${pad(now.completionTokens.toLocaleString(), 8)}${ref((run1?.completionTokens ?? 0).toLocaleString())}`,
    `wall time        ${pad(fmtDur(now.wallMs), 8)}${ref(fmtDur(run1?.wallMs ?? 0))}`,
    `est. cost        ${pad(fmtUsd(now.usd), 8)}${ref(fmtUsd(run1?.usd ?? 0))}`,
  ];
  return [
    muted(`┌─ LEDGER ${line.slice(9)}┐`),
    ...rows.map((r) => `  ${r}`),
    muted(`└${line}┘`),
  ].join("\n");
}

export function savingsReport(cmp: RunComparison): string {
  const line = "─".repeat(60);
  const head = `  ${pad("metric", 26)}${pad("run 1 (cold)", 16)}${pad("run 2 (cached)", 16)}run 3 (semantic)`;
  const body = cmp.rows.map(
    (r) =>
      `  ${pad(r.metric, 26)}${pad(r.run1, 16)}${pad(green(r.run2), 16 + (green("x").length - 1))}${r.run3}`,
  );
  return [
    rust(`┌─ SAVINGS REPORT ${line.slice(17)}┐`),
    bold(head),
    ...body,
    rust(`└${line}┘`),
  ].join("\n");
}

if (process.argv.includes("--demo")) {
  console.log(
    header({
      task: "add per-API-key rate limiting (60/min) + tests",
      run: 2,
      mode: "warm",
      providers: ["openai", "anthropic", "gemini"],
      degraded: [],
    }),
  );
  console.log();
  console.log(stageCard(1, "RETRIEVE", "done", "12 chunks · arch.png"));
  console.log(stageCard(2, "PLAN", "hit", "exact · 0 ms billed"));
  console.log(stageCard(3, "BRIEF", "hit", "3 workers -> 1 upstream call (coalesced)"));
  console.log(stageCard(4, "IMPLEMENT", "running", "worker 2/3  src/rateLimit.ts"));
  console.log(stageCard(5, "REVIEW", "pending", ""));
  console.log();
  console.log(
    colorizeDiff(
      "--- a/src/rateLimit.ts\n+++ b/src/rateLimit.ts\n@@ -0,0 +1,3 @@\n+export function rateLimit(cfg) {\n+  return true;\n+}\n unchanged line",
    ),
  );
  console.log();
  console.log(
    ledgerPanel(
      { upstreamCalls: 0, promptTokens: 0, completionTokens: 0, usd: 0, wallMs: 2100 },
      {
        upstreamCalls: 11,
        promptTokens: 48210,
        completionTokens: 3940,
        usd: 0.71,
        wallMs: 192000,
      },
    ),
  );
  console.log();
  console.log(
    savingsReport({
      rows: [
        { metric: "upstream calls", run1: "11", run2: "0", run3: "2" },
        { metric: "est. cost", run1: "$0.71", run2: "$0.00", run3: "$0.06" },
        { metric: "wall time", run1: "3m 12s", run2: "2.1 s", run3: "8.0 s" },
      ],
    }),
  );
}

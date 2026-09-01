import { appendFileSync, readFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type Surface =
  | "chat_completions"
  | "messages"
  | "embeddings"
  | "image_embeddings";

export interface TraceEvent {
  t: number;
  run: 1 | 2 | 3;
  stage: string;
  type: "chat" | "messages" | "embeddings" | "image_embeddings" | "test" | "note";
  provider: string;
  model: string;
  surface: Surface | null;
  hit: boolean;
  hitKind: "exact" | "semantic" | null;
  semanticScore: number | null;
  promptTokens: number;
  completionTokens: number;
  billedPromptTokens: number;
  billedCompletionTokens: number;
  latencyMs: number;
  usd: number;
  coalesced: boolean;
  note: string;
}

// Illustrative list prices, USD per 1M tokens [prompt, completion].
// Same posture as the /dashboard: clearly not a billing oracle.
const PRICES: Record<string, [number, number]> = {
  "gpt-4o-mini": [0.15, 0.6],
  "gpt-4o": [2.5, 10],
  "claude-sonnet-5": [3, 15],
  "claude-opus-5": [15, 75],
  "text-embedding-3-small": [0.02, 0],
  "text-embedding-3-large": [0.13, 0],
  "gemini-embedding-001": [0, 0],
};

export function priceUsd(
  model: string,
  promptTokens: number,
  completionTokens: number,
): number {
  const p = PRICES[model];
  if (!p) return 0;
  return (promptTokens / 1e6) * p[0] + (completionTokens / 1e6) * p[1];
}

export class Trace {
  private start = Date.now();
  constructor(
    private runFilePath: string,
    private run: 1 | 2 | 3,
  ) {
    mkdirSync(dirname(runFilePath), { recursive: true });
    appendFileSync(runFilePath, "", { flag: "w" });
  }
  add(ev: Omit<TraceEvent, "t" | "run">): void {
    const full: TraceEvent = { ...ev, t: Date.now() - this.start, run: this.run };
    try {
      appendFileSync(this.runFilePath, JSON.stringify(full) + "\n");
    } catch (e) {
      console.error("trace write failed (non-fatal):", (e as Error).message);
    }
  }
  close(): void {}
}

function read(jsonlPath: string): TraceEvent[] {
  return readFileSync(jsonlPath, "utf8")
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as TraceEvent);
}

export interface RunSummary {
  upstreamCalls: number;
  promptTokens: number;
  completionTokens: number;
  billedPromptTokens: number;
  billedCompletionTokens: number;
  wallMs: number;
  usd: number;
  hits: number;
  misses: number;
  coalescedCalls: number;
  semanticHits: number;
}

export function summarize(jsonlPath: string): RunSummary {
  const evs = read(jsonlPath).filter(
    (e) => e.type !== "note" && e.type !== "test",
  );
  const s: RunSummary = {
    upstreamCalls: 0,
    promptTokens: 0,
    completionTokens: 0,
    billedPromptTokens: 0,
    billedCompletionTokens: 0,
    wallMs: evs.length ? Math.max(...evs.map((e) => e.t)) : 0,
    usd: 0,
    hits: 0,
    misses: 0,
    coalescedCalls: 0,
    semanticHits: 0,
  };
  for (const e of evs) {
    s.promptTokens += e.promptTokens;
    s.completionTokens += e.completionTokens;
    s.billedPromptTokens += e.billedPromptTokens;
    s.billedCompletionTokens += e.billedCompletionTokens;
    s.usd += e.usd;
    if (e.hit) s.hits++;
    else s.misses++;
    if (e.hitKind === "semantic") s.semanticHits++;
    if (e.coalesced) s.coalescedCalls++;
    else if (!e.hit) s.upstreamCalls++;
  }
  return s;
}

function fmtUsd(n: number): string {
  return "$" + n.toFixed(n < 1 ? 4 : 2);
}
function fmtDur(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export interface RunComparison {
  rows: Array<{ metric: string; run1: string; run2: string; run3: string }>;
}

export function compare(
  run1: string,
  run2: string,
  run3: string,
): RunComparison {
  const a = summarize(run1);
  const b = summarize(run2);
  const c = summarize(run3);
  const row = (
    metric: string,
    f: (s: RunSummary) => string,
  ): RunComparison["rows"][number] => ({
    metric,
    run1: f(a),
    run2: f(b),
    run3: f(c),
  });
  return {
    rows: [
      row("upstream calls", (s) => String(s.upstreamCalls)),
      row("prompt tokens billed", (s) => s.billedPromptTokens.toLocaleString()),
      row("completion tokens billed", (s) =>
        s.billedCompletionTokens.toLocaleString()),
      row("wall time", (s) => fmtDur(s.wallMs)),
      row("est. cost", (s) => fmtUsd(s.usd)),
      row("cache hits", (s) => String(s.hits)),
      row("semantic hits", (s) => String(s.semanticHits)),
      row("coalesced calls", (s) => String(s.coalescedCalls)),
    ],
  };
}

import { header, stageCard, ledgerPanel, type HeaderOpts } from "./render.ts";
import type { StageOutcome } from "./orchestrator.ts";
import type { Ledger } from "./ledger.ts";

const STAGE_NAMES = [
  "RETRIEVE",
  "PLAN",
  "BRIEF",
  "IMPLEMENT",
  "REVIEW",
  "FIX",
  "VERIFY",
];

interface LedgerView {
  upstreamCalls: number;
  promptTokens: number;
  completionTokens: number;
  usd: number;
  wallMs: number;
}

/** An in-place redrawing board: header (static) + 7 stage lines + a ledger. */
export class Board {
  private drawnLines = 0;
  private start = Date.now();
  private stream = "";

  constructor(
    private opts: HeaderOpts,
    private ledger: Ledger,
    private run1?: LedgerView,
  ) {
    process.stdout.write(header(opts) + "\n\n");
  }

  setStream(text: string): void {
    this.stream = text.replace(/\s+/g, " ").slice(-70);
  }

  draw(stages: StageOutcome[]): void {
    const cols = (process.stdout.columns ?? 80) - 1;
    const lines: string[] = [];
    for (let i = 0; i < STAGE_NAMES.length; i++) {
      const s = stages[i];
      const name = STAGE_NAMES[i]!;
      if (!s) {
        lines.push(stageCard(i + 1, name, "pending", ""));
      } else {
        let detail = s.detail;
        if (name === "REVIEW" && s.status === "done" && this.stream)
          detail = this.stream;
        const status =
          s.status === "failed"
            ? "running"
            : s.status === "hit"
              ? "hit"
              : "done";
        lines.push(
          stageCard(
            i + 1,
            name,
            s.status === "failed" ? "running" : status,
            s.status === "failed" ? `FAILED — ${detail}` : detail,
          ),
        );
      }
    }
    const snap = this.ledger.snapshot();
    const view: LedgerView = { ...snap, wallMs: Date.now() - this.start };
    lines.push("");
    lines.push(...ledgerPanel(view, this.run1).split("\n"));

    const clipped = lines.map((l) => clip(l, cols));
    if (this.drawnLines > 0)
      process.stdout.write(`\x1b[${this.drawnLines}A\x1b[0J`);
    process.stdout.write(clipped.join("\n") + "\n");
    this.drawnLines = clipped.length;
  }
}

/** Clip a possibly-ANSI-colored string to n visible columns. */
function clip(s: string, n: number): string {
  let visible = 0;
  let out = "";
  let i = 0;
  while (i < s.length) {
    if (s[i] === "\x1b") {
      const end = s.indexOf("m", i);
      if (end === -1) break;
      out += s.slice(i, end + 1);
      i = end + 1;
      continue;
    }
    if (visible >= n) {
      i++;
      continue;
    }
    out += s[i];
    visible++;
    i++;
  }
  return out;
}

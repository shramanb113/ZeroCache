export interface LedgerEvent {
  billedPromptTokens: number;
  billedCompletionTokens: number;
  usd: number;
  hit: boolean;
  coalesced: boolean;
}

export interface LedgerSnapshot {
  upstreamCalls: number;
  promptTokens: number;
  completionTokens: number;
  usd: number;
}

export class Ledger {
  private s: LedgerSnapshot = {
    upstreamCalls: 0,
    promptTokens: 0,
    completionTokens: 0,
    usd: 0,
  };

  addEvent(e: LedgerEvent): void {
    this.s.promptTokens += e.billedPromptTokens;
    this.s.completionTokens += e.billedCompletionTokens;
    this.s.usd += e.usd;
    if (!e.hit && !e.coalesced) this.s.upstreamCalls += 1;
  }

  snapshot(): LedgerSnapshot {
    return { ...this.s };
  }
}

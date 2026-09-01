import { Trace, priceUsd, type Surface } from "./trace.ts";

export interface ZcResult<T> {
  data: T;
  hit: boolean;
  hitKind: "exact" | "semantic" | null;
  semanticScore: number | null;
  hitsHeader: number | null;
  missesHeader: number | null;
  latencyMs: number;
  billedPromptTokens: number;
  billedCompletionTokens: number;
  usd: number;
  coalesced: boolean;
}

interface CallMeta {
  stage: string;
  coalesced?: boolean;
}

function num(h: Headers, name: string): number | null {
  const v = h.get(name);
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export class ZerocacheClient {
  private baseUrl: string;
  private trace: Trace;

  constructor(opts: { baseUrl: string; trace: Trace }) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, "");
    this.trace = opts.trace;
  }

  private url(provider: string, path: string): string {
    return `${this.baseUrl}/${provider}/v1/${path}`;
  }

  async chat(
    provider: string,
    apiKey: string,
    body: object,
    meta: CallMeta,
  ): Promise<ZcResult<Record<string, unknown>>> {
    const t0 = performance.now();
    const res = await fetch(this.url(provider, "chat/completions"), {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const latencyMs = Math.round(performance.now() - t0);
    const json = (await res.json()) as Record<string, unknown>;
    if (!res.ok) {
      throw new Error(
        `chat ${provider} -> ${res.status}: ${JSON.stringify(json).slice(0, 300)}`,
      );
    }
    const hit = res.headers.get("x-zerocache-completion-hit") === "true";
    const hitKind = (res.headers.get("x-zerocache-completion-hit-kind") ??
      null) as "exact" | "semantic" | null;
    const semanticScore = num(res.headers, "x-zerocache-semantic-score");
    const usage = (json.usage ?? {}) as {
      prompt_tokens?: number;
      completion_tokens?: number;
    };
    const model = (body as { model?: string }).model ?? "";
    const coalesced = meta.coalesced ?? false;
    const billed = this.record({
      stage: meta.stage,
      type: "chat",
      surface: "chat_completions",
      provider,
      model,
      hit,
      hitKind,
      semanticScore,
      rawPrompt: usage.prompt_tokens ?? 0,
      rawCompletion: usage.completion_tokens ?? 0,
      coalesced,
      latencyMs,
    });
    return {
      data: json,
      hit,
      hitKind,
      semanticScore,
      hitsHeader: null,
      missesHeader: null,
      latencyMs,
      ...billed,
      coalesced,
    };
  }

  async messagesStream(
    provider: string,
    apiKey: string,
    body: object,
    onDelta: (text: string) => void,
    meta: CallMeta,
  ): Promise<ZcResult<{ text: string; raw: unknown }>> {
    const t0 = performance.now();
    const res = await fetch(this.url(provider, "messages"), {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
        accept: "text/event-stream",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const txt = await res.text().catch(() => "");
      throw new Error(`messages ${provider} -> ${res.status}: ${txt.slice(0, 300)}`);
    }
    const hit = res.headers.get("x-zerocache-completion-hit") === "true";
    const hitKind = (res.headers.get("x-zerocache-completion-hit-kind") ??
      null) as "exact" | "semantic" | null;
    const semanticScore = num(res.headers, "x-zerocache-semantic-score");

    let text = "";
    let inputTokens = 0;
    let outputTokens = 0;
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of res.body as AsyncIterable<Uint8Array>) {
      buf += decoder.decode(chunk, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop() ?? "";
      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === "[DONE]" || payload === "") continue;
          let ev: Record<string, unknown>;
          try {
            ev = JSON.parse(payload);
          } catch {
            continue;
          }
          const type = ev.type as string | undefined;
          if (type === "content_block_delta") {
            const delta = ev.delta as { text?: string } | undefined;
            if (delta?.text) {
              text += delta.text;
              onDelta(delta.text);
            }
          } else if (type === "message_start") {
            const msg = ev.message as
              | { usage?: { input_tokens?: number } }
              | undefined;
            inputTokens = msg?.usage?.input_tokens ?? inputTokens;
          } else if (type === "message_delta") {
            const u = ev.usage as { output_tokens?: number } | undefined;
            outputTokens = u?.output_tokens ?? outputTokens;
          }
        }
      }
    }
    const latencyMs = Math.round(performance.now() - t0);
    const model = (body as { model?: string }).model ?? "";
    const coalesced = meta.coalesced ?? false;
    const billed = this.record({
      stage: meta.stage,
      type: "messages",
      surface: "messages",
      provider,
      model,
      hit,
      hitKind,
      semanticScore,
      rawPrompt: inputTokens,
      rawCompletion: outputTokens,
      coalesced,
      latencyMs,
    });
    return {
      data: { text, raw: {} },
      hit,
      hitKind,
      semanticScore,
      hitsHeader: null,
      missesHeader: null,
      latencyMs,
      ...billed,
      coalesced,
    };
  }

  async embed(
    provider: string,
    apiKey: string,
    model: string,
    input: string[],
    meta: CallMeta,
  ): Promise<ZcResult<number[][]>> {
    return this.embedInner(provider, apiKey, model, input, "embeddings", meta);
  }

  async embedImages(
    provider: string,
    apiKey: string,
    model: string,
    dataUris: string[],
    meta: CallMeta,
  ): Promise<ZcResult<number[][]>> {
    return this.embedInner(
      provider,
      apiKey,
      model,
      dataUris,
      "images/embeddings",
      meta,
    );
  }

  private async embedInner(
    provider: string,
    apiKey: string,
    model: string,
    input: string[],
    path: "embeddings" | "images/embeddings",
    meta: CallMeta,
  ): Promise<ZcResult<number[][]>> {
    const t0 = performance.now();
    const res = await fetch(this.url(provider, path), {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, input }),
    });
    const latencyMs = Math.round(performance.now() - t0);
    const json = (await res.json()) as {
      data?: { embedding: number[]; index: number }[];
      usage?: { prompt_tokens?: number };
    };
    if (!res.ok) {
      throw new Error(
        `embed ${provider} -> ${res.status}: ${JSON.stringify(json).slice(0, 300)}`,
      );
    }
    const hitsHeader = num(res.headers, "x-zerocache-hits");
    const missesHeader = num(res.headers, "x-zerocache-misses");
    const hit = missesHeader === 0 && (hitsHeader ?? 0) > 0;
    const vectors = (json.data ?? [])
      .slice()
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
    const surface: Surface =
      path === "images/embeddings" ? "image_embeddings" : "embeddings";
    const coalesced = meta.coalesced ?? false;
    const billed = this.record({
      stage: meta.stage,
      type: surface === "image_embeddings" ? "image_embeddings" : "embeddings",
      surface,
      provider,
      model,
      hit,
      hitKind: hit ? "exact" : null,
      semanticScore: null,
      rawPrompt: json.usage?.prompt_tokens ?? 0,
      rawCompletion: 0,
      coalesced,
      latencyMs,
    });
    return {
      data: vectors,
      hit,
      hitKind: hit ? "exact" : null,
      semanticScore: null,
      hitsHeader,
      missesHeader,
      latencyMs,
      ...billed,
      coalesced,
    };
  }

  private record(r: {
    stage: string;
    type: "chat" | "messages" | "embeddings" | "image_embeddings";
    surface: Surface;
    provider: string;
    model: string;
    hit: boolean;
    hitKind: "exact" | "semantic" | null;
    semanticScore: number | null;
    rawPrompt: number;
    rawCompletion: number;
    coalesced: boolean;
    latencyMs: number;
  }): { billedPromptTokens: number; billedCompletionTokens: number; usd: number } {
    const free = r.hit || r.coalesced;
    const billedPromptTokens = free ? 0 : r.rawPrompt;
    const billedCompletionTokens = free ? 0 : r.rawCompletion;
    const usd = priceUsd(r.model, billedPromptTokens, billedCompletionTokens);
    this.trace.add({
      stage: r.stage,
      type: r.type,
      provider: r.provider,
      model: r.model,
      surface: r.surface,
      hit: r.hit,
      hitKind: r.hitKind,
      semanticScore: r.semanticScore,
      promptTokens: r.rawPrompt,
      completionTokens: r.rawCompletion,
      billedPromptTokens,
      billedCompletionTokens,
      latencyMs: r.latencyMs,
      usd,
      coalesced: r.coalesced,
      note: "",
    });
    return { billedPromptTokens, billedCompletionTokens, usd };
  }
}

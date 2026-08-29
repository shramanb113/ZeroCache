import { priceFor, type PriceOverrides } from "./prices";

interface Sample {
  labels: Record<string, string>;
  value: number;
}

/** Parse Prometheus text exposition format into { metricName: Sample[] }. */
export function parsePrometheus(text: string): Record<string, Sample[]> {
  const out: Record<string, Sample[]> = Object.create(null);
  for (const line of text.split("\n")) {
    if (!line || line[0] === "#") continue;
    const m = line.match(/^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)\s*$/);
    if (!m) continue;
    const value = Number(m[3]);
    if (!Number.isFinite(value)) continue;
    const labels: Record<string, string> = {};
    if (m[2]) {
      const re = /([a-zA-Z_]\w*)="((?:[^"\\]|\\.)*)"/g;
      let lm: RegExpExecArray | null;
      while ((lm = re.exec(m[2]))) {
        labels[lm[1]] = lm[2].replace(/\\"/g, '"').replace(/\\\\/g, "\\");
      }
    }
    (out[m[1]] ||= []).push({ labels, value });
  }
  return out;
}

function sumByProvider(series: Sample[] | undefined, provider: string): number {
  let s = 0;
  for (const row of series ?? []) if (row.labels.provider === provider) s += row.value;
  return s;
}

export interface ProviderRow {
  provider: string;
  /** completion cache */
  cHit: number;
  cSemanticHit: number;
  cMiss: number;
  promptTokensSaved: number;
  completionTokensSaved: number;
  cRate: number | null;
  /** embedding cache */
  eHit: number;
  eMiss: number;
  estEmbedTokensSaved: number;
  eRate: number | null;
  /** dollars */
  completionUsd: number;
  embeddingUsd: number;
}

export interface Snapshot {
  rows: ProviderRow[];
  totalUsd: number;
  completionUsd: number;
  embeddingUsd: number;
  servedFromCache: number;
  overallHitRate: number | null;
  completionHitRate: number | null;
  embeddingHitRate: number | null;
  completionTokensSaved: number;
  embeddingTokensSaved: number;
  completionSemanticHits: number;
}

const M = {
  cHit: "zerocache_completion_cache_hits_total",
  cMiss: "zerocache_completion_cache_misses_total",
  cSemantic: "zerocache_completion_semantic_hits_total",
  cPromptSaved: "zerocache_completion_prompt_tokens_saved_total",
  cComplSaved: "zerocache_completion_completion_tokens_saved_total",
  eHit: "zerocache_cache_hits_total",
  eMiss: "zerocache_cache_misses_total",
  eBilled: "zerocache_provider_prompt_tokens_total",
} as const;

/** The metric family names this dashboard depends on, exported so the Rust side
 *  can assert the embedded page still references the current names. */
export const DEPENDS_ON_METRICS: string[] = Object.values(M);

export function shape(
  raw: Record<string, Sample[]>,
  overrides: PriceOverrides,
  assumedEmbedTokens: number,
): Snapshot {
  const providers = new Set<string>();
  for (const key in raw) {
    for (const row of raw[key]) if (row.labels.provider) providers.add(row.labels.provider);
  }

  const rows: ProviderRow[] = [];
  let completionUsd = 0;
  let embeddingUsd = 0;
  let cHit = 0;
  let cSemantic = 0;
  let cMiss = 0;
  let eHit = 0;
  let eMiss = 0;
  let promptTokensSaved = 0;
  let completionTokensSaved = 0;
  let embeddingTokensSaved = 0;

  for (const p of [...providers].sort()) {
    const pcHit = sumByProvider(raw[M.cHit], p);
    const pcSemantic = sumByProvider(raw[M.cSemantic], p);
    const pcMiss = sumByProvider(raw[M.cMiss], p);
    const pPromptSaved = sumByProvider(raw[M.cPromptSaved], p);
    const pComplSaved = sumByProvider(raw[M.cComplSaved], p);
    const peHit = sumByProvider(raw[M.eHit], p);
    const peMiss = sumByProvider(raw[M.eMiss], p);
    const peBilled = sumByProvider(raw[M.eBilled], p);

    const price = priceFor(p, overrides);
    const compUsd = (pPromptSaved / 1e6) * price.in + (pComplSaved / 1e6) * price.out;
    const avgTokPerMiss = peMiss > 0 ? peBilled / peMiss : assumedEmbedTokens;
    const estEmbedTokens = peHit * avgTokPerMiss;
    const embUsd = (estEmbedTokens / 1e6) * price.embed;

    rows.push({
      provider: p,
      cHit: pcHit,
      cSemanticHit: pcSemantic,
      cMiss: pcMiss,
      promptTokensSaved: pPromptSaved,
      completionTokensSaved: pComplSaved,
      cRate: pcHit + pcMiss > 0 ? pcHit / (pcHit + pcMiss) : null,
      eHit: peHit,
      eMiss: peMiss,
      estEmbedTokensSaved: estEmbedTokens,
      eRate: peHit + peMiss > 0 ? peHit / (peHit + peMiss) : null,
      completionUsd: compUsd,
      embeddingUsd: embUsd,
    });

    completionUsd += compUsd;
    embeddingUsd += embUsd;
    cHit += pcHit;
    cSemantic += pcSemantic;
    cMiss += pcMiss;
    eHit += peHit;
    eMiss += peMiss;
    promptTokensSaved += pPromptSaved;
    completionTokensSaved += pComplSaved;
    embeddingTokensSaved += estEmbedTokens;
  }

  const servedFromCache = cHit + eHit;
  const totalReq = servedFromCache + cMiss + eMiss;

  return {
    rows,
    totalUsd: completionUsd + embeddingUsd,
    completionUsd,
    embeddingUsd,
    servedFromCache,
    overallHitRate: totalReq > 0 ? servedFromCache / totalReq : null,
    completionHitRate: cHit + cMiss > 0 ? cHit / (cHit + cMiss) : null,
    embeddingHitRate: eHit + eMiss > 0 ? eHit / (eHit + eMiss) : null,
    completionTokensSaved: promptTokensSaved + completionTokensSaved,
    embeddingTokensSaved,
    completionSemanticHits: cSemantic,
  };
}

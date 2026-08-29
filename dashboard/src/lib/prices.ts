export interface Price {
  in: number;
  out: number;
  embed: number;
}

/** Illustrative list prices, USD per 1M tokens. Not authoritative -- the UI lets
 *  the operator override any of them, persisted to localStorage. */
export const DEFAULT_PRICES: Record<string, Price> = {
  default: { in: 0.5, out: 1.5, embed: 0.02 },
  openai: { in: 0.15, out: 0.6, embed: 0.02 },
  gemini: { in: 0.1, out: 0.4, embed: 0.15 },
  groq: { in: 0.1, out: 0.3, embed: 0.02 },
  deepseek: { in: 0.14, out: 0.28, embed: 0.02 },
  mistral: { in: 0.2, out: 0.6, embed: 0.1 },
  together: { in: 0.2, out: 0.6, embed: 0.02 },
  openrouter: { in: 0.5, out: 1.5, embed: 0.02 },
  xai: { in: 2.0, out: 10.0, embed: 0.02 },
  fireworks: { in: 0.2, out: 0.8, embed: 0.02 },
  huggingface: { in: 0.2, out: 0.6, embed: 0.02 },
};

export const DEFAULT_ASSUMED_EMBED_TOKENS = 50;

const PRICE_KEY = "zc-dash-prices-v1";
const ASSUMED_KEY = "zc-dash-assumed-embed-v1";
const THEME_KEY = "zc-dash-theme-v1";

export type PriceOverrides = Record<string, Partial<Price>>;

export function loadOverrides(): PriceOverrides {
  try {
    const v = JSON.parse(localStorage.getItem(PRICE_KEY) || "null");
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

export function saveOverrides(o: PriceOverrides): void {
  try {
    localStorage.setItem(PRICE_KEY, JSON.stringify(o));
  } catch {
    /* private mode / disabled storage -- run without persistence */
  }
}

export function loadAssumedEmbedTokens(): number {
  try {
    const v = Number(localStorage.getItem(ASSUMED_KEY));
    return v > 0 ? v : DEFAULT_ASSUMED_EMBED_TOKENS;
  } catch {
    return DEFAULT_ASSUMED_EMBED_TOKENS;
  }
}

export function saveAssumedEmbedTokens(n: number): void {
  try {
    localStorage.setItem(ASSUMED_KEY, String(n));
  } catch {
    /* ignore */
  }
}

export function priceFor(provider: string, overrides: PriceOverrides): Price {
  const seed = DEFAULT_PRICES[provider] ?? DEFAULT_PRICES.default;
  const o = overrides[provider] ?? {};
  return {
    in: Number.isFinite(o.in) ? (o.in as number) : seed.in,
    out: Number.isFinite(o.out) ? (o.out as number) : seed.out,
    embed: Number.isFinite(o.embed) ? (o.embed as number) : seed.embed,
  };
}

export type Theme = "light" | "dark" | "system";

export function loadTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function saveTheme(t: Theme): void {
  try {
    if (t === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, t);
  } catch {
    /* ignore */
  }
}

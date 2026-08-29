import { useMemo } from "react";
import {
  DEFAULT_ASSUMED_EMBED_TOKENS,
  DEFAULT_PRICES,
  priceFor,
  saveAssumedEmbedTokens,
  saveOverrides,
  type PriceOverrides,
} from "../lib/prices";

interface Props {
  providersSeen: string[];
  overrides: PriceOverrides;
  assumedEmbedTokens: number;
  onOverrides: (o: PriceOverrides) => void;
  onAssumed: (n: number) => void;
}

type Field = "in" | "out" | "embed";

export default function PricingPanel({
  providersSeen,
  overrides,
  assumedEmbedTokens,
  onOverrides,
  onAssumed,
}: Props) {
  const names = useMemo(() => {
    const s = new Set(providersSeen);
    for (const k of Object.keys(DEFAULT_PRICES)) if (k !== "default") s.add(k);
    return [...s].sort();
  }, [providersSeen]);

  function edit(provider: string, field: Field, raw: string) {
    const n = Number(raw);
    const next: PriceOverrides = { ...overrides, [provider]: { ...overrides[provider] } };
    if (Number.isFinite(n) && n >= 0) next[provider][field] = n;
    else delete next[provider][field];
    if (Object.keys(next[provider]).length === 0) delete next[provider];
    onOverrides(next);
    saveOverrides(next);
  }

  function reset() {
    onOverrides({});
    saveOverrides({});
    onAssumed(DEFAULT_ASSUMED_EMBED_TOKENS);
    saveAssumedEmbedTokens(DEFAULT_ASSUMED_EMBED_TOKENS);
  }

  return (
    <details className="assump">
      <summary>Pricing assumptions</summary>
      <div className="a-body">
        <p className="note">
          Illustrative list prices, USD per 1M tokens — edit for your model and tier.
          Completion savings are exact (the stored response carries the token counts the
          provider did not bill). Embedding savings are estimated as hits × average tokens
          per observed miss, or × the assumed size below when no miss has been seen yet.
          Saved in this browser only.
        </p>
        <div className="tablewrap">
          <table className="grid" style={{ minWidth: 520 }}>
            <thead>
              <tr>
                <th className="name">Provider</th>
                <th>Input $/Mtok</th>
                <th>Output $/Mtok</th>
                <th>Embedding $/Mtok</th>
              </tr>
            </thead>
            <tbody>
              {names.map((p) => {
                const price = priceFor(p, overrides);
                return (
                  <tr key={p}>
                    <td className="name">{p}</td>
                    {(["in", "out", "embed"] as Field[]).map((f) => (
                      <td key={f}>
                        <input
                          type="number"
                          min={0}
                          step={0.01}
                          defaultValue={price[f]}
                          onChange={(e) => edit(p, f, e.target.value)}
                          aria-label={`${p} ${f} price per million tokens`}
                        />
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="a-actions">
          <label className="a-hint">
            Assumed tokens per embedding (no miss seen yet):{" "}
            <input
              type="number"
              min={1}
              step={1}
              value={assumedEmbedTokens}
              onChange={(e) => {
                const n = Number(e.target.value);
                const v = n > 0 ? n : DEFAULT_ASSUMED_EMBED_TOKENS;
                onAssumed(v);
                saveAssumedEmbedTokens(v);
              }}
              style={{ width: 70 }}
            />
          </label>
          <button className="ghost" onClick={reset}>
            Reset to defaults
          </button>
        </div>
      </div>
    </details>
  );
}

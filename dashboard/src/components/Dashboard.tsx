import { useEffect, useMemo, useState } from "react";
import { useMetrics } from "../lib/useMetrics";
import { compact, pct, usd } from "../lib/format";
import { loadTheme, saveTheme, type Theme } from "../lib/prices";
import SavingsChart from "./SavingsChart";
import Sparkline from "./Sparkline";
import HitRate from "./HitRate";
import ProviderTable from "./ProviderTable";
import PricingPanel from "./PricingPanel";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>("system");
  useEffect(() => setTheme(loadTheme()), []);
  const cycle = () => {
    const next: Theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system";
    setTheme(next);
    saveTheme(next);
    if (next === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = next;
  };
  return [theme, cycle];
}

const blend = (a: number | null, b: number | null): number | null => {
  const parts = [a, b].filter((x): x is number => x !== null);
  return parts.length ? parts.reduce((x, y) => x + y, 0) / parts.length : null;
};

export default function Dashboard() {
  const m = useMetrics();
  const [theme, cycleTheme] = useTheme();

  const s = m.snapshot;
  const since = s && m.baseline !== null ? Math.max(0, s.totalUsd - m.baseline) : 0;

  const usdSeries = useMemo(() => m.history.map((h) => h.totalUsd), [m.history]);
  const rateSeries = useMemo(
    () => m.history.map((h) => blend(h.completionHitRate, h.embeddingHitRate)),
    [m.history],
  );

  const rateDelta = useMemo(() => {
    if (m.history.length < 2) return null;
    const a = blend(m.history[0].completionHitRate, m.history[0].embeddingHitRate);
    const b = blend(
      m.history[m.history.length - 1].completionHitRate,
      m.history[m.history.length - 1].embeddingHitRate,
    );
    return a !== null && b !== null ? (b - a) * 100 : null;
  }, [m.history]);

  const statusText =
    m.paused
      ? "paused"
      : m.status === "error"
        ? "disconnected"
        : m.lastUpdated
          ? "updated " + new Date(m.lastUpdated).toLocaleTimeString()
          : "starting…";
  const statusKind = m.paused ? "stale" : m.status === "error" ? "error" : "ok";

  return (
    <>
      <header className="bar">
        <h1>Zerocache</h1>
        <span className="sub">cache savings, live</span>
        <span className="spacer" />
        <div className="controls">
          <span className="dot" data-kind={statusKind} />
          <span className="status">{statusText}</span>
          <button className="ghost" onClick={() => m.setPaused(!m.paused)}>
            {m.paused ? "Resume" : "Pause"}
          </button>
          <button className="ghost" onClick={cycleTheme}>
            Theme: {theme}
          </button>
        </div>
      </header>

      {m.status === "error" && (
        <div className="banner">
          Can’t reach <code>/metrics</code> — {m.error}. Showing the last snapshot.
        </div>
      )}

      <div className="stack" data-stale={m.status === "error"}>
        <section className="hero">
          <div className="label">Estimated cost avoided since this Zerocache started</div>
          <div className="fig">{usd(s?.totalUsd ?? 0)}</div>
          <div className="delta">
            {m.history.length >= 2 ? `+${usd(since)} since this page loaded` : "collecting…"}
          </div>
          <div className="caption">
            {usd(s?.completionUsd ?? 0)} completions (measured) · {usd(s?.embeddingUsd ?? 0)} embeddings
            (estimated)
          </div>
        </section>

        <section className="kpis">
          <Tile
            label="Requests served from cache"
            value={compact(s?.servedFromCache ?? 0)}
            spark={usdSeries}
            delta={m.history.length >= 2 ? `+${usd(since)} value since page load` : "collecting…"}
            up={since > 0}
          />
          <Tile
            label="Overall hit rate"
            value={pct(s?.overallHitRate ?? null)}
            spark={rateSeries}
            delta={
              rateDelta === null
                ? "since page load"
                : `${rateDelta >= 0 ? "+" : ""}${rateDelta.toFixed(1)} pts since page load`
            }
            up={rateDelta !== null && rateDelta > 0.05}
          />
          <Tile
            label="Completion tokens saved"
            value={compact(s?.completionTokensSaved ?? 0)}
            spark={usdSeries}
            delta="measured from stored responses"
            up={(s?.completionTokensSaved ?? 0) > 0}
          />
          <Tile
            label="Est. embedding tokens saved"
            value={compact(s?.embeddingTokensSaved ?? 0)}
            spark={usdSeries}
            delta="estimated · see assumptions"
            up={false}
          />
        </section>

        <section className="row2">
          <SavingsChart history={m.history} />
          <HitRate
            completionHitRate={s?.completionHitRate ?? null}
            embeddingHitRate={s?.embeddingHitRate ?? null}
            cHits={sumRows(s?.rows, "cHit")}
            cTotal={sumRows(s?.rows, "cHit") + sumRows(s?.rows, "cMiss")}
            eHits={sumRows(s?.rows, "eHit")}
            eTotal={sumRows(s?.rows, "eHit") + sumRows(s?.rows, "eMiss")}
          />
        </section>

        <ProviderTable rows={s?.rows ?? []} />

        <PricingPanel
          providersSeen={(s?.rows ?? []).map((r) => r.provider)}
          overrides={m.overrides}
          assumedEmbedTokens={m.assumedEmbedTokens}
          onOverrides={m.setOverrides}
          onAssumed={m.setAssumedEmbedTokens}
        />

        <footer className="note">
          <p>
            Polls <code>/metrics</code> every 2s. Counters are cumulative since this Zerocache
            process started and reset to zero on restart — this page detects that and clears the
            session chart.
          </p>
          <p>
            Completion savings are exact: the stored response carries the token counts the provider
            did not bill on a hit. Embedding savings are an estimate — hits × average tokens per
            observed miss (or × the assumed size when no miss has been seen), × your embedding price.
          </p>
          <p>
            {s && s.rows.length
              ? `${s.rows.length} provider${s.rows.length === 1 ? "" : "s"} active.`
              : "Waiting for the first cached request."}
          </p>
        </footer>
      </div>
    </>
  );
}

function sumRows(
  rows: { cHit: number; cMiss: number; eHit: number; eMiss: number }[] | undefined,
  key: "cHit" | "cMiss" | "eHit" | "eMiss",
): number {
  return (rows ?? []).reduce((s, r) => s + r[key], 0);
}

interface TileProps {
  label: string;
  value: string;
  spark: (number | null)[];
  delta: string;
  up: boolean;
}

function Tile({ label, value, spark, delta, up }: TileProps) {
  return (
    <div className="tile">
      <div className="t-label">{label}</div>
      <div className="t-value">{value}</div>
      <Sparkline values={spark} />
      <div className="t-delta" data-up={up}>
        {delta}
      </div>
    </div>
  );
}

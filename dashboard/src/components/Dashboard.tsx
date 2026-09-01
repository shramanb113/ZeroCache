import { useEffect, useMemo, useState } from "react";
import { useMetrics } from "../lib/useMetrics";
import { useCountUp } from "../lib/useCountUp";
import { compact, pct, usd2 } from "../lib/format";
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
  const total = useCountUp(s?.totalUsd ?? 0);

  // split the (animated) total by the real measured/estimated proportion,
  // rounding in cents so the two rows always sum to the hero figure exactly
  const measuredShare = s && s.totalUsd > 0 ? s.completionUsd / s.totalUsd : 0;
  const totalCents = Math.round(total * 100);
  const measuredCents = Math.round(totalCents * measuredShare);
  const measuredUsd = measuredCents / 100;
  const estimatedUsd = (totalCents - measuredCents) / 100;

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

  const startedAt = useMemo(
    () => (m.history.length ? new Date(m.history[0].t) : null),
    [m.history],
  );

  const statusText = m.paused
    ? "paused"
    : m.status === "error"
      ? "disconnected"
      : m.lastUpdated
        ? new Date(m.lastUpdated).toLocaleTimeString(undefined, {
            hour12: false,
          })
        : "starting…";
  const statusKind = m.paused ? "stale" : m.status === "error" ? "error" : "ok";

  return (
    <>
      <header className="bar">
        <h1>zerocache</h1>
        <span className="rule" />
        <span className="sub">savings monitor</span>
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
        <section className="panel hero reveal">
          <div className="eyebrow">
            Cost avoided
            <span className="trailing">
              {startedAt
                ? "since " + startedAt.toLocaleString(undefined, { hour12: false })
                : "since this instance started"}
            </span>
          </div>

          <div className="readout">
            <div className="fig">{usd2(total)}</div>
            <div className="delta" data-up={since > 0}>
              {m.history.length >= 2 ? `▲ +${usd2(since)} this session` : "collecting…"}
            </div>
          </div>

          <div className="split">
            <div className="srow">
              <span>
                <span className="tag">measured</span> completions
              </span>
              <b>{usd2(measuredUsd)}</b>
            </div>
            <div className="sbar">
              <div className="sfill" style={{ transform: `scaleX(${measuredShare})` }} />
            </div>
            <div className="srow">
              <span>
                <span className="tag">estimated</span> embeddings
              </span>
              <b>{usd2(estimatedUsd)}</b>
            </div>
          </div>

          {s && s.completionSemanticHits > 0 && (
            <div className="caption">
              {compact(s.completionSemanticHits)} completion hits matched via the semantic tier
            </div>
          )}
        </section>

        <section className="panel cluster reveal">
          <Cell
            label="Served from cache"
            value={compact(s?.servedFromCache ?? 0)}
            spark={usdSeries}
            delta={m.history.length >= 2 ? `+${usd2(since)} value` : "collecting…"}
            up={since > 0}
          />
          <Cell
            label="Overall hit rate"
            value={pct(s?.overallHitRate ?? null)}
            spark={rateSeries}
            delta={
              rateDelta === null || Math.abs(rateDelta) < 0.05
                ? "since page load"
                : `${rateDelta >= 0 ? "+" : ""}${rateDelta.toFixed(1)} pts`
            }
            up={rateDelta !== null && rateDelta > 0.05}
          />
          <Cell
            label="Completion tokens saved"
            value={compact(s?.completionTokensSaved ?? 0)}
            caption="measured from stored responses"
          />
          <Cell
            label="Est. embedding tokens saved"
            value={compact(s?.embeddingTokensSaved ?? 0)}
            caption="estimated · see assumptions"
          />
        </section>

        <section className="row2 reveal">
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

        <div className="reveal">
          <ProviderTable rows={s?.rows ?? []} />
        </div>

        <div className="reveal">
          <PricingPanel
            providersSeen={(s?.rows ?? []).map((r) => r.provider)}
            overrides={m.overrides}
            assumedEmbedTokens={m.assumedEmbedTokens}
            onOverrides={m.setOverrides}
            onAssumed={m.setAssumedEmbedTokens}
          />
        </div>

        <footer className="note reveal">
          <p>
            Polls <code>/metrics</code> every 2s. Counters are cumulative since this Zerocache
            process started and reset to zero on restart — this page detects that and clears the
            session chart.
          </p>
          <p>
            Completion savings are exact: the stored response carries the token counts the provider
            did not bill on a hit. Embedding savings are an estimate — hits × average tokens per
            observed miss (or × the assumed size when no miss has been seen), × your embedding
            price.
          </p>
          <p>
            {s && s.rows.length
              ? `${s.rows.length} provider${s.rows.length === 1 ? "" : "s"} active.`
              : "Waiting for the first cached request."}
          </p>
          {s && s.crossReplicaCoalesced > 0 && (
            <p>
              {compact(s.crossReplicaCoalesced)} request
              {s.crossReplicaCoalesced === 1 ? "" : "s"} served from a peer replica's fill — upstream
              calls avoided across replicas.
            </p>
          )}
          {s && s.semanticIndexEventsApplied > 0 && (
            <p>
              {compact(s.semanticIndexEventsApplied)} semantic-index change-feed events applied on
              this replica (multi-replica redis backend).
            </p>
          )}
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

interface CellProps {
  label: string;
  value: string;
  spark?: (number | null)[];
  delta?: string;
  caption?: string;
  up?: boolean;
}

function Cell({ label, value, spark, delta, caption, up }: CellProps) {
  return (
    <div className="cell">
      <div className="c-label">{label}</div>
      <div className="c-value">{value}</div>
      {spark && <Sparkline values={spark} />}
      {delta && (
        <div className="c-delta" data-up={up}>
          {delta}
        </div>
      )}
      {caption && <div className="c-delta">{caption}</div>}
    </div>
  );
}

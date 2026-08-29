import { int, pct } from "../lib/format";

interface MeterProps {
  label: string;
  rate: number | null;
  hits: number;
  total: number;
  soft?: boolean;
}

function Meter({ label, rate, hits, total, soft }: MeterProps) {
  const value = rate === null ? undefined : Math.round(rate * 1000) / 10;
  return (
    <div
      className="meter"
      role="meter"
      aria-label={`${label} cache hit rate`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value}
    >
      <div className="m-top">
        <span>{label}</span>
        <b>{pct(rate)}</b>
      </div>
      <div className="m-track">
        <div
          className="m-fill"
          data-soft={soft ? "true" : undefined}
          style={{ transform: `scaleX(${rate ?? 0})` }}
        />
      </div>
      <div className="m-sub">
        {total > 0 ? `${int(hits)} of ${int(total)} requests` : "no requests yet"}
      </div>
    </div>
  );
}

interface Props {
  completionHitRate: number | null;
  embeddingHitRate: number | null;
  cHits: number;
  cTotal: number;
  eHits: number;
  eTotal: number;
}

export default function HitRate({
  completionHitRate,
  embeddingHitRate,
  cHits,
  cTotal,
  eHits,
  eTotal,
}: Props) {
  return (
    <div className="panel card">
      <div className="eyebrow">Hit rate</div>
      <div className="sub">share of requests served without a provider call</div>
      <Meter label="Completions" rate={completionHitRate} hits={cHits} total={cTotal} />
      <Meter label="Embeddings" rate={embeddingHitRate} hits={eHits} total={eTotal} soft />
    </div>
  );
}

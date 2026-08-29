import { int, pct } from "../lib/format";

interface MeterProps {
  label: string;
  rate: number | null;
  hits: number;
  total: number;
}

function Meter({ label, rate, hits, total }: MeterProps) {
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
        <div className="m-fill" style={{ width: `${(rate ?? 0) * 100}%` }} />
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
    <div className="card">
      <h2>Hit rate</h2>
      <div className="h2sub">share of requests served without a provider call</div>
      <Meter label="Completions" rate={completionHitRate} hits={cHits} total={cTotal} />
      <Meter label="Embeddings" rate={embeddingHitRate} hits={eHits} total={eTotal} />
    </div>
  );
}

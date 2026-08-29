import type { ProviderRow } from "../lib/metrics";
import { compact, int, pct, usd } from "../lib/format";

interface Props {
  rows: ProviderRow[];
}

function Num({ value, format }: { value: number; format: (n: number) => string }) {
  return <td className={value ? undefined : "zero"}>{format(value)}</td>;
}

export default function ProviderTable({ rows }: Props) {
  return (
    <div className="card">
      <h2>By provider</h2>
      <div className="h2sub">
        cumulative counters from <code>/metrics</code>
      </div>
      <div className="tablewrap">
        <table className="grid">
          <thead>
            <tr>
              <th className="name">Provider</th>
              <th>Compl. hits</th>
              <th>Compl. misses</th>
              <th>Hit rate</th>
              <th>Tokens saved</th>
              <th>Embed hits</th>
              <th>Embed misses</th>
              <th>Hit rate</th>
              <th>Est. tokens saved</th>
              <th>Cost avoided</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td className="empty" colSpan={10}>
                  No requests recorded yet — send one through{" "}
                  <code>/{"{provider}"}/v1/chat/completions</code> or{" "}
                  <code>/{"{provider}"}/v1/embeddings</code>.
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr key={r.provider}>
                  <td className="name">{r.provider}</td>
                  <Num value={r.cHit} format={int} />
                  <Num value={r.cMiss} format={int} />
                  <td className={r.cRate === null ? "zero" : undefined}>{pct(r.cRate)}</td>
                  <Num value={r.promptTokensSaved + r.completionTokensSaved} format={compact} />
                  <Num value={r.eHit} format={int} />
                  <Num value={r.eMiss} format={int} />
                  <td className={r.eRate === null ? "zero" : undefined}>{pct(r.eRate)}</td>
                  <Num value={Math.round(r.estEmbedTokensSaved)} format={compact} />
                  <td className="save">{usd(r.completionUsd + r.embeddingUsd)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

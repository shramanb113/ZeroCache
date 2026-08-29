import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistoryPoint } from "../lib/useMetrics";
import { clock, usd } from "../lib/format";

interface Props {
  history: HistoryPoint[];
}

interface Row {
  elapsed: number;
  total: number;
  completion: number;
  embedding: number;
}

function TooltipBody({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload || !payload.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rc-tooltip">
      <div className="rc-v">{usd(r.total)}</div>
      <div className="rc-row">at {clock(r.elapsed)}</div>
      <div className="rc-row">
        <span className="rc-key" />
        completions {usd(r.completion)}
      </div>
      <div className="rc-row">
        <span className="rc-key" style={{ background: "var(--deemph)" }} />
        embeddings {usd(r.embedding)}
      </div>
    </div>
  );
}

export default function SavingsChart({ history }: Props) {
  const [asTable, setAsTable] = useState(false);

  const rows: Row[] = useMemo(() => {
    if (!history.length) return [];
    const t0 = history[0].t;
    return history.map((h) => ({
      elapsed: h.t - t0,
      total: h.totalUsd,
      completion: h.completionUsd,
      embedding: h.embeddingUsd,
    }));
  }, [history]);

  return (
    <div className="card">
      <h2>Cost avoided over this session</h2>
      <div className="h2sub">
        cumulative, since the page loaded &middot;{" "}
        <a role="button" tabIndex={0} onClick={() => setAsTable((v) => !v)} onKeyDown={(e) => e.key === "Enter" && setAsTable((v) => !v)}>
          {asTable ? "chart" : "table"}
        </a>
      </div>

      {rows.length < 2 ? (
        <div style={{ height: 200, display: "grid", placeItems: "center", color: "var(--muted)", fontSize: 12 }}>
          collecting samples…
        </div>
      ) : asTable ? (
        <div className="tablewrap">
          <table className="grid">
            <thead>
              <tr>
                <th className="name">Elapsed</th>
                <th>Cost avoided</th>
                <th>Completions</th>
                <th>Embeddings</th>
              </tr>
            </thead>
            <tbody>
              {sampleRows(rows).map((r) => (
                <tr key={r.elapsed}>
                  <td className="name">{clock(r.elapsed)}</td>
                  <td>{usd(r.total)}</td>
                  <td>{usd(r.completion)}</td>
                  <td>{usd(r.embedding)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ width: "100%", height: 200 }}>
          <ResponsiveContainer>
            <AreaChart data={rows} margin={{ top: 12, right: 16, bottom: 4, left: 4 }}>
              <defs>
                <linearGradient id="savingsFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--series)" stopOpacity={0.22} />
                  <stop offset="100%" stopColor="var(--series)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="elapsed"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v) => clock(v as number)}
                tick={{ fontSize: 11 }}
                stroke="var(--axis)"
                tickLine={false}
              />
              <YAxis
                width={72}
                domain={[0, "auto"]}
                tickFormatter={(v) => usd(v as number)}
                tick={{ fontSize: 11 }}
                stroke="var(--axis)"
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<TooltipBody />} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} />
              <Area
                type="monotone"
                dataKey="total"
                stroke="var(--series)"
                strokeWidth={2}
                fill="url(#savingsFill)"
                isAnimationActive={false}
                dot={false}
                activeDot={{ r: 4, fill: "var(--series)", stroke: "var(--surface)", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/** cap the table view at ~20 evenly-spaced rows */
function sampleRows(rows: Row[]): Row[] {
  const step = Math.max(1, Math.floor(rows.length / 20));
  const out: Row[] = [];
  for (let i = 0; i < rows.length; i += step) out.push(rows[i]);
  const last = rows[rows.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

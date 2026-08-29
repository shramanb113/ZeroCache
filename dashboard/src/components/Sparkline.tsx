import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";

interface Props {
  values: (number | null)[];
}

/** A bare 12-point trend for a stat tile: de-emphasis line, no axes, no tooltip.
 *  The number beside it carries the actual value. */
export default function Sparkline({ values }: Props) {
  const clean = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (clean.length < 2) return <div className="spark" />;
  const data = clean.slice(-12).map((v, i) => ({ i, v }));

  return (
    <div className="spark">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 2, right: 1, bottom: 2, left: 1 }}>
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <Area
            type="monotone"
            dataKey="v"
            stroke="var(--deemph)"
            strokeWidth={2}
            fill="var(--deemph)"
            fillOpacity={0.14}
            isAnimationActive={false}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

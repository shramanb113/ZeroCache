export function usd(n: number): string {
  const a = Math.abs(n);
  if (a === 0) return "$0.00";
  if (a >= 1000) return "$" + n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 1) return "$" + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (a >= 0.01) return "$" + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  // sub-cent: significant digits so tiny values (and adjacent axis ticks) stay distinct
  return "$" + n.toLocaleString(undefined, { maximumSignificantDigits: 2 });
}

export function int(n: number): string {
  return Math.round(n).toLocaleString();
}

export function compact(n: number): string {
  if (n >= 1e6) return (n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "M";
  if (n >= 1e4) return (n / 1e3).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "K";
  return int(n);
}

export function pct(r: number | null): string {
  return r === null ? "—" : (r * 100).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "%";
}

/** Elapsed milliseconds as m:ss. */
export function clock(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

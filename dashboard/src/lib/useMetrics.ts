import { useCallback, useEffect, useRef, useState } from "react";
import { parsePrometheus, shape, type Snapshot } from "./metrics";
import {
  loadAssumedEmbedTokens,
  loadOverrides,
  type PriceOverrides,
} from "./prices";

/** Absolute -- the dashboard is served under /dashboard but /metrics is at the
 *  origin root. Do not make this relative. */
const METRICS_URL = "/metrics";
const POLL_MS = 2000;
const HISTORY_CAP = 300;

export interface HistoryPoint {
  t: number;
  totalUsd: number;
  completionUsd: number;
  embeddingUsd: number;
  completionHitRate: number | null;
  embeddingHitRate: number | null;
}

export interface MetricsState {
  snapshot: Snapshot | null;
  history: HistoryPoint[];
  /** total $ at the first successful poll after this page loaded */
  baseline: number | null;
  status: "starting" | "ok" | "error";
  error: string | null;
  lastUpdated: number | null;
  paused: boolean;
  overrides: PriceOverrides;
  assumedEmbedTokens: number;
}

export interface MetricsApi extends MetricsState {
  setPaused: (p: boolean) => void;
  setOverrides: (o: PriceOverrides) => void;
  setAssumedEmbedTokens: (n: number) => void;
  refreshNow: () => void;
}

export function useMetrics(): MetricsApi {
  const [overrides, setOverrides] = useState<PriceOverrides>({});
  const [assumedEmbedTokens, setAssumedEmbedTokens] = useState<number>(50);
  const [paused, setPaused] = useState(false);

  const [state, setState] = useState<
    Pick<MetricsState, "snapshot" | "history" | "baseline" | "status" | "error" | "lastUpdated">
  >({
    snapshot: null,
    history: [],
    baseline: null,
    status: "starting",
    error: null,
    lastUpdated: null,
  });

  // last raw /metrics body, so a price/assumption edit recomputes immediately
  const lastTextRef = useRef<string | null>(null);
  const overridesRef = useRef(overrides);
  const assumedRef = useRef(assumedEmbedTokens);
  overridesRef.current = overrides;
  assumedRef.current = assumedEmbedTokens;

  // hydrate persisted settings on mount (client only)
  useEffect(() => {
    setOverrides(loadOverrides());
    setAssumedEmbedTokens(loadAssumedEmbedTokens());
  }, []);

  const applyText = useCallback((text: string) => {
    lastTextRef.current = text;
    const next = shape(parsePrometheus(text), overridesRef.current, assumedRef.current);
    setState((prev) => {
      // a cumulative counter going backwards means the server restarted
      const reset = prev.snapshot !== null && next.totalUsd + 1e-9 < prev.snapshot.totalUsd;
      const history = reset ? [] : prev.history.slice();
      history.push({
        t: Date.now(),
        totalUsd: next.totalUsd,
        completionUsd: next.completionUsd,
        embeddingUsd: next.embeddingUsd,
        completionHitRate: next.completionHitRate,
        embeddingHitRate: next.embeddingHitRate,
      });
      if (history.length > HISTORY_CAP) history.shift();
      return {
        snapshot: next,
        history,
        baseline: reset || prev.baseline === null ? next.totalUsd : prev.baseline,
        status: "ok",
        error: null,
        lastUpdated: Date.now(),
      };
    });
  }, []);

  const recompute = useCallback(() => {
    if (lastTextRef.current === null) return;
    const next = shape(parsePrometheus(lastTextRef.current), overridesRef.current, assumedRef.current);
    setState((prev) => ({ ...prev, snapshot: next }));
  }, []);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(METRICS_URL, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      applyText(await res.text());
    } catch (err) {
      setState((prev) => ({
        ...prev,
        status: "error",
        error: err instanceof Error ? err.message : "unreachable",
      }));
    }
  }, [applyText]);

  // recompute when pricing inputs change, without waiting for the next poll
  useEffect(() => {
    recompute();
  }, [overrides, assumedEmbedTokens, recompute]);

  // polling loop
  useEffect(() => {
    if (paused) return;
    let alive = true;
    void poll();
    const id = setInterval(() => {
      if (alive) void poll();
    }, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [paused, poll]);

  return {
    ...state,
    paused,
    overrides,
    assumedEmbedTokens,
    setPaused,
    setOverrides,
    setAssumedEmbedTokens,
    refreshNow: () => void poll(),
  };
}

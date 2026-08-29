# Zerocache savings dashboard

A live, single-page dashboard that shows how much a running Zerocache instance
has saved — cache hit rates, tokens not billed, and an estimated dollar figure —
by polling `GET /metrics` every 2 seconds.

Built with **Astro + React islands + Recharts**. The build output
(`dist/`) is **committed** and embedded into `zerocache-http` at compile time
(`include_dir!` in `zerocache-http/src/dashboard.rs`), so a plain
`cargo build -p zerocache-http` needs no Node toolchain.

## Using it

```sh
cargo run -p zerocache-http
# then open the URL it logs:
#   savings dashboard at http://localhost:8080/dashboard
```

It is served on the same origin as `/metrics`, so there is no CORS setup and
nothing to configure. The page is unauthenticated, like `/metrics`, `/health`
and `/ready`.

## What it shows

- **Hero** — estimated total cost avoided since the process started.
- **KPI row** — requests served from cache, overall hit rate, completion tokens
  saved, estimated embedding tokens saved (each with a session sparkline and a
  since-page-load delta).
- **Session chart** — cumulative cost avoided over the time the page has been
  open, with a table-view toggle.
- **Hit-rate meters** — completions and embeddings separately.
- **By-provider table** — the raw cumulative counters, per provider.
- **Pricing assumptions** — editable $/Mtok per provider (input / output /
  embedding), persisted to `localStorage`. Recompute is instant, no poll wait.

### How savings are computed

- **Completions** are exact: the stored response carries the `usage` block, so a
  hit knows precisely which prompt/completion tokens were not billed. Dollars =
  `tokens_saved / 1e6 × your price`. When the opt-in semantic tier is enabled the
  hero caption also breaks out how many completion hits came from a semantic
  near-match (`zerocache_completion_semantic_hits_total`, a subset of
  `zerocache_completion_cache_hits_total`).
- **Embeddings** are an estimate — the metrics only expose tokens *billed* on
  misses, not a per-hit token count. The dashboard uses
  `hits × (average tokens per observed miss)`, or `hits × the assumed size` in
  the pricing panel when no miss has been seen yet.

Counters reset to zero when the Zerocache process restarts; the page detects the
backwards jump and clears the session chart.

## Developing

```sh
cd dashboard
npm install
npm run dev        # http://localhost:4321/dashboard, proxied metrics won't work —
                   # run against a real instance for live data, or use `preview`
npm run build      # regenerate dist/ — commit it
npm run check      # astro + tsc type check
```

After any change under `dashboard/src/`, run `npm run build` and commit the new
`dist/`. `zerocache-http`'s `bundle_references_every_metric_name_it_parses` test
fails if the built bundle stops referencing a metric name the Rust side emits.

Structure:

| Path | Role |
| --- | --- |
| `src/pages/index.astro` | shell; mounts the island with `client:only="react"` |
| `src/components/Dashboard.tsx` | orchestrator — header, hero, KPI row, layout |
| `src/components/SavingsChart.tsx` | Recharts area chart + table toggle |
| `src/components/{Sparkline,HitRate,ProviderTable,PricingPanel}.tsx` | the pieces |
| `src/lib/metrics.ts` | Prometheus-text parser + per-provider savings shaping |
| `src/lib/useMetrics.ts` | polling loop, session history, reset detection |
| `src/lib/prices.ts` | default prices + `localStorage` helpers |
| `src/lib/format.ts` | number/currency/clock formatters |

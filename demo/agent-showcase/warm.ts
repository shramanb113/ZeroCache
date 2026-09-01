import { config as loadEnv } from "dotenv";
import { executeRun } from "./run.ts";

loadEnv();

const BASE = process.env.ZEROCACHE_BASE_URL ?? "http://localhost:8080";

async function semanticEnabled(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/metrics`);
    return (await r.text()).includes("zerocache_completion_semantic_hits_total");
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  if (!process.env.SHOWCASE_API_KEY && !process.env.OPENAI_API_KEY) {
    console.error(
      "OPENAI_API_KEY (or SHOWCASE_API_KEY) is required (see .env.example).",
    );
    process.exit(1);
  }
  try {
    const h = await fetch(`${BASE}/health`);
    if (!h.ok) throw new Error("not healthy");
  } catch {
    console.error(
      `Zerocache not answering at ${BASE}/health. Start: cargo run -p zerocache-http`,
    );
    process.exit(1);
  }

  const runs: (1 | 2 | 3)[] = [1, 2];
  if (await semanticEnabled()) runs.push(3);

  for (const run of runs) {
    process.stdout.write(`warming run ${run}... `);
    const { result } = await executeRun(
      { run, check: false, record: false },
      { quiet: true },
    );
    console.log(
      `${result.testsPassed} passed / ${result.testsFailed} failed`,
    );
  }
  console.log("cache warmed — record with:  npm run -- run --run=2");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

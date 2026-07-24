import { LibSQLVector } from "@mastra/libsql";

// Local, file-backed vector store (no external DB server) -- confirmed against
// @mastra/libsql's own embedded docs (node_modules/@mastra/libsql/dist/docs/references/reference-vectors-libsql.md)
// and its type definitions (node_modules/@mastra/libsql/dist/vector/index.d.ts), not from memory.
const DB_URL = process.env.ZEROCACHE_DEMO_DB_URL ?? "file:rag-vector-store.db";

const store = new LibSQLVector({ id: "aurora-rag-vector-store", url: DB_URL });

// Text (OpenAI text-embedding-3-small, 1536-dim) and image (Gemini gemini-embedding-2,
// up to 3072-dim) vectors live in two different, mutually incomparable embedding spaces --
// different models, different dimensionality, no shared geometry. LibSQL's createIndex()
// requires one fixed dimension per index (confirmed: doCreateIndex validates
// `Number.isInteger(dimension) && dimension > 0` and creates an `F32_BLOB(dimension)`
// column at that fixed width), so a single shared index literally cannot hold both
// vector shapes correctly. Using two separate indexes -- one per kind -- therefore
// enforces the kind separation structurally (a text query vector can never even reach
// the table holding image vectors), on top of the metadata.kind filter passed through
// for defense in depth.
const TEXT_INDEX = "aurora_text";
const IMAGE_INDEX = "aurora_image";
const TEXT_DIMENSION = 1536;
const IMAGE_DIMENSION = 3072;

function indexNameForKind(kind: "text" | "image"): string {
  return kind === "text" ? TEXT_INDEX : IMAGE_INDEX;
}

export async function createIndex(): Promise<void> {
  // CREATE TABLE IF NOT EXISTS under the hood -- safe to call on every ingest run.
  await store.createIndex({ indexName: TEXT_INDEX, dimension: TEXT_DIMENSION });
  await store.createIndex({ indexName: IMAGE_INDEX, dimension: IMAGE_DIMENSION });
}

export async function upsert(
  entries: { id: string; vector: number[]; metadata: Record<string, unknown> }[],
): Promise<void> {
  const byKind = new Map<"text" | "image", { id: string; vector: number[]; metadata: Record<string, unknown> }[]>();
  for (const entry of entries) {
    const kind = entry.metadata.kind as "text" | "image";
    if (kind !== "text" && kind !== "image") {
      throw new Error(`upsert entry ${entry.id} is missing a valid metadata.kind ("text" | "image")`);
    }
    const bucket = byKind.get(kind) ?? [];
    bucket.push(entry);
    byKind.set(kind, bucket);
  }

  for (const [kind, group] of byKind) {
    // `ids` matching an existing vector_id perform a true upsert (ON CONFLICT(vector_id)
    // DO UPDATE), not an insert-duplicate -- confirmed against the compiled adapter
    // source, node_modules/@mastra/libsql/dist/index.js. This is what makes ingesting the
    // same filename twice with new content update the entry rather than duplicate it.
    await store.upsert({
      indexName: indexNameForKind(kind),
      vectors: group.map((entry) => entry.vector),
      metadata: group.map((entry) => entry.metadata),
      ids: group.map((entry) => entry.id),
    });
  }
}

export async function query(
  vector: number[],
  topK: number,
  filter: { kind: "text" | "image" },
): Promise<{ id: string; score: number; metadata: Record<string, unknown> }[]> {
  const results = await store.query({
    indexName: indexNameForKind(filter.kind),
    queryVector: vector,
    topK,
    // Redundant with the index-level separation above, but kept as an explicit,
    // native metadata filter so the `kind` invariant doesn't silently depend on
    // index routing alone if this store is ever refactored to a single shared index.
    filter: { kind: filter.kind },
  });

  return results.map((result) => ({
    id: result.id,
    score: result.score,
    metadata: result.metadata ?? {},
  }));
}

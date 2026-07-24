// A minimal RAG pipeline: chunk a small corpus, embed the chunks through
// Zerocache (not directly against OpenAI), store the vectors in an
// in-memory store, then answer a query via similarity search.
//
// Run it twice in a row (`npm run rag`) to see the point of the whole
// exercise: the first run is all misses, the second run -- same corpus,
// same model, same key -- should be all hits, and noticeably faster.

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { MODEL, ZEROCACHE_OPENAI_BASE_URL, requireApiKey } from "./zerocache-config.js";
import { loggingFetch } from "./logging-fetch.js";

const CORPUS = [
  {
    title: "Zerocache overview",
    text: "Zerocache is a Rust-native embedding cache that sits between an application's ingestion pipeline and its embedding provider. It intercepts OpenAI-compatible /v1/embeddings requests, serves previously-computed vectors from a local content-addressed store, and forwards only cache misses upstream. Adoption requires no SDK -- a consumer just points its existing embedding client at Zerocache's base_url.",
  },
  {
    title: "Cache key design",
    text: "The cache key is derived from blake3(owner_id, provider, model, model_version, text). Owner_id is a hash of the caller's forwarded API key, never the raw key itself. Including model_version means a model upgrade can never silently return a stale-but-plausible vector from an older model version.",
  },
  {
    title: "Multi-tenant isolation",
    text: "Every request requires a real forwarded provider API key via the Authorization header. Two different callers embedding identical text under the same model share a cache hit for free, but two different callers never share a cache entry with each other, by design -- this avoids both unfair cost sharing and a cache-timing existence-leak risk.",
  },
  {
    title: "Storage backends",
    text: "Zerocache ships two EmbeddingStore implementations: an embedded sled store for local development and single-instance deployments, and a Redis-backed store for multi-replica Kubernetes deployments. Selection happens at startup via the ZEROCACHE_STORAGE_BACKEND environment variable.",
  },
  {
    title: "Supported providers",
    text: "Three EmbeddingProvider adapters exist today: OpenAI, Mistral, and Gemini, selected per-request via the URL path as POST /{provider}/v1/embeddings. Each adapter chunks its input into batches of at most 100 texts per upstream call.",
  },
];

async function ingest(vectorStore: MemoryVectorStore) {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });
  const docs = await splitter.createDocuments(
    CORPUS.map((c) => c.text),
    CORPUS.map((c) => ({ title: c.title })),
  );
  console.log(`chunked ${CORPUS.length} source documents into ${docs.length} chunks`);

  const start = performance.now();
  await vectorStore.addDocuments(docs);
  console.log(`ingested ${docs.length} chunks in ${(performance.now() - start).toFixed(0)}ms\n`);
}

async function query(vectorStore: MemoryVectorStore, question: string) {
  console.log(`\nquery: "${question}"`);
  const results = await vectorStore.similaritySearch(question, 2);
  for (const doc of results) {
    console.log(`  - [${doc.metadata.title}] ${doc.pageContent.slice(0, 80)}...`);
  }
}

async function main() {
  const apiKey = requireApiKey();

  const embeddings = new OpenAIEmbeddings({
    apiKey,
    model: MODEL,
    configuration: {
      baseURL: ZEROCACHE_OPENAI_BASE_URL,
      fetch: loggingFetch("rag"),
    },
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  console.log("=== ingesting corpus (expect all misses on a cold cache) ===");
  await ingest(vectorStore);

  await query(vectorStore, "How does Zerocache decide which cache entry belongs to which caller?");
  await query(vectorStore, "What storage backends does Zerocache support?");
}

main().catch((err) => {
  console.error("RAG pipeline failed:", err);
  process.exitCode = 1;
});

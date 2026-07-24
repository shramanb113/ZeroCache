import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { embedText } from "../zerocache/client";
import { requireEnv, ZEROCACHE_BASE_URL } from "../zerocache/env";
import { query } from "../rag/vector-store";

// Deliberately NOT importing from "../rag/ingest" -- that module calls
// requireEnv("OPENAI_API_KEY")/requireEnv("GEMINI_API_KEY") at module load
// time, so importing it here would throw at import time if either var is
// unset, even for a run that only ever calls this text-only tool.

export const searchDocumentsTool = createTool({
  id: "searchDocuments",
  description:
    "Search the Aurora Cloud Storage knowledge base by text query. Use this for questions about pricing, " +
    "features, authentication, troubleshooting, policies, or anything answerable from written documentation. " +
    "Do not use this for questions about the content of an image.",
  inputSchema: z.object({
    query: z.string().describe("The text search query"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        id: z.string(),
        score: z.number(),
        file: z.string().optional(),
        text: z.string().optional(),
      }),
    ),
  }),
  execute: async ({ query: searchQuery }) => {
    const apiKey = requireEnv("OPENAI_API_KEY");

    const { embeddings } = await embedText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey,
      model: "text-embedding-3-small",
      input: searchQuery,
    });

    // topK: 3 is deliberately smaller than a typical RAG default -- with only
    // 5-6 text documents in the corpus, a question spanning more than 3
    // documents' worth of information genuinely cannot be answered from one
    // call, which is what makes multi-hop follow-up calls a real necessity
    // rather than a coin flip.
    const results = await query(embeddings[0], 3, { kind: "text" });

    return {
      results: results.map((result) => ({
        id: result.id,
        score: result.score,
        file: typeof result.metadata.file === "string" ? result.metadata.file : undefined,
        text: typeof result.metadata.text === "string" ? result.metadata.text : undefined,
      })),
    };
  },
});

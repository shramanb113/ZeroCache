import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { embedImage } from "../zerocache/client";
import { query } from "../rag/vector-store";

// Deliberately NOT importing from "../rag/ingest" -- see search-documents.ts
// for why (module-load-time requireEnv() calls in ingest.ts would throw at
// import time for an unrelated code path).
const ZEROCACHE_BASE_URL = process.env.ZEROCACHE_BASE_URL ?? "http://localhost:8080";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`missing required env var ${name}`);
  return value;
}

export const searchImagesTool = createTool({
  id: "searchImages",
  description:
    "Search the Aurora Cloud Storage knowledge base by image similarity. Use this when the user provides " +
    "an image and asks what it shows or which stored image it resembles. Do not use this for text-only questions.",
  inputSchema: z.object({
    imageBase64: z.string().describe("Base64-encoded image data (no data:<mime>;base64, prefix)"),
    imageMimeType: z.string().describe("The image's MIME type, e.g. image/png"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        id: z.string(),
        score: z.number(),
        file: z.string().optional(),
      }),
    ),
  }),
  execute: async ({ imageBase64, imageMimeType }) => {
    const apiKey = requireEnv("GEMINI_API_KEY");

    const { embeddings } = await embedImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey,
      model: "gemini-embedding-2",
      images: [{ mimeType: imageMimeType, base64: imageBase64 }],
    });

    // topK: 2 -- only 2 images exist in the corpus, so this returns
    // everything; the job here is discriminating between them, not
    // narrowing a large set.
    const results = await query(embeddings[0], 2, { kind: "image" });

    return {
      results: results.map((result) => ({
        id: result.id,
        score: result.score,
        file: typeof result.metadata.file === "string" ? result.metadata.file : undefined,
      })),
    };
  },
});

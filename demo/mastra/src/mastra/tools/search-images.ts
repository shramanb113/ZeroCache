import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { embedImage } from "../zerocache/client";
import { requireEnv, ZEROCACHE_BASE_URL } from "../zerocache/env";
import { query } from "../rag/vector-store";

// Deliberately NOT importing from "../rag/ingest" -- see search-documents.ts
// for why (module-load-time requireEnv() calls in ingest.ts would throw at
// import time for an unrelated code path).

// The image's bytes are read from `requestContext`, NOT from a model-generated
// tool-call argument -- found via Task 12's battle test (a real bug, not a
// hypothetical): asking the LLM to retype a base64 image blob verbatim as a
// tool argument is unreliable even for a small (~600-char) test image --
// gpt-4o-mini reproducibly truncated/corrupted it (624 -> 600 chars, mismatch)
// on repeated runs, causing Zerocache's embed call to fail on the mangled
// payload. `RequestContext` (@mastra/core/request-context, verified against
// node_modules/@mastra/core/dist/tools/types.d.ts and
// dist/docs/references/reference-agents-generate.md's `options.requestContext`
// entry) is Mastra's real, current mechanism for exactly this: injecting
// per-call data a tool can read directly in `execute()`, bypassing model
// generation entirely. The caller populates requestContext with the image
// before invoking `ragAgent.generate(prompt, { requestContext })`; the model
// still decides *whether* to call searchImages (that's the agentic part,
// unaffected), it just never has to transcribe the bytes itself.
export interface SearchImagesRequestContext {
  imageBase64: string;
  imageMimeType: string;
}

export const searchImagesTool = createTool({
  id: "searchImages",
  description:
    "Search the Aurora Cloud Storage knowledge base by image similarity, using the image attached to this " +
    "request. Use this when the user's message asks what an attached image shows or which stored image it " +
    "resembles. Do not use this for text-only questions, and do not invent image data yourself.",
  inputSchema: z.object({}),
  outputSchema: z.object({
    results: z.array(
      z.object({
        id: z.string(),
        score: z.number(),
        file: z.string().optional(),
      }),
    ),
  }),
  execute: async (_input, { requestContext }) => {
    const imageBase64 = requestContext.get("imageBase64") as string | undefined;
    const imageMimeType = requestContext.get("imageMimeType") as string | undefined;
    if (!imageBase64 || !imageMimeType) {
      throw new Error(
        "searchImages was called but no image was attached to this request " +
          "(requestContext is missing imageBase64/imageMimeType)",
      );
    }

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

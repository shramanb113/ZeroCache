import { createStep, createWorkflow } from '@mastra/core/workflows';
import { ModelRouterEmbeddingModel } from '@mastra/core/llm';
import { embedMany } from 'ai';
import { z } from 'zod';

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const embedTexts = createStep({
  id: 'embed-texts',
  description: 'Embeds each input text through Zerocache (gemini provider), timing the call',
  inputSchema: z.object({
    apiKey: z.string().describe('Your Gemini API key'),
    texts: z.array(z.string()).min(2).describe('Texts to check for near-duplicates'),
  }),
  outputSchema: z.object({
    texts: z.array(z.string()),
    embeddings: z.array(z.array(z.number())),
    elapsedMs: z.number(),
  }),
  execute: async ({ inputData }) => {
    const { apiKey, texts } = inputData;

    const embedder = new ModelRouterEmbeddingModel({
      id: 'openai/gemini-embedding-001',
      url: 'http://127.0.0.1:8080/gemini/v1',
      apiKey,
    });

    const start = Date.now();
    const result = await embedMany({ model: embedder, values: texts });
    const elapsedMs = Date.now() - start;

    return { texts, embeddings: result.embeddings, elapsedMs };
  },
});

const findDuplicates = createStep({
  id: 'find-duplicates',
  description: 'Finds near-duplicate text pairs by cosine similarity',
  inputSchema: z.object({
    texts: z.array(z.string()),
    embeddings: z.array(z.array(z.number())),
    elapsedMs: z.number(),
  }),
  outputSchema: z.object({
    elapsedMs: z.number(),
    duplicatePairs: z.array(
      z.object({ textA: z.string(), textB: z.string(), similarity: z.number() }),
    ),
  }),
  execute: async ({ inputData }) => {
    const { texts, embeddings, elapsedMs } = inputData;
    const THRESHOLD = 0.92;
    const duplicatePairs: { textA: string; textB: string; similarity: number }[] = [];

    for (let i = 0; i < texts.length; i++) {
      for (let j = i + 1; j < texts.length; j++) {
        const similarity = cosineSimilarity(embeddings[i], embeddings[j]);
        if (similarity >= THRESHOLD) {
          duplicatePairs.push({ textA: texts[i], textB: texts[j], similarity });
        }
      }
    }

    return { elapsedMs, duplicatePairs };
  },
});

export const duplicateFinderWorkflow = createWorkflow({
  id: 'duplicate-finder',
  description:
    'Embeds a batch of texts through Zerocache and finds near-duplicates by cosine similarity. Run it twice with an overlapping batch to see the cache speedup show up in elapsedMs.',
  inputSchema: z.object({
    apiKey: z.string(),
    texts: z.array(z.string()).min(2),
  }),
  outputSchema: z.object({
    elapsedMs: z.number(),
    duplicatePairs: z.array(
      z.object({ textA: z.string(), textB: z.string(), similarity: z.number() }),
    ),
  }),
})
  .then(embedTexts)
  .then(findDuplicates)
  .commit();

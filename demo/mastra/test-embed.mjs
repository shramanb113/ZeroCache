import { embedMany } from 'ai';
import { ModelRouterEmbeddingModel } from '@mastra/core/llm';

const embedder = new ModelRouterEmbeddingModel({
  id: 'openai/gemini-embedding-001',
  url: 'http://127.0.0.1:8080/gemini/v1',
  apiKey: process.env.ZEROCACHE_DEMO_KEY,
});

async function run(label) {
  try {
    const result = await embedMany({
      model: embedder,
      values: ['hello from mastra via zerocache, gemini demo run'],
    });
    console.log(`[${label}] SUCCESS — embedding length:`, result.embeddings[0].length);
  } catch (err) {
    console.log(`[${label}] ERROR MESSAGE:`, err?.message ?? String(err));
    if (err?.cause) console.log(`[${label}] CAUSE:`, err.cause);
    if (err?.url) console.log(`[${label}] REQUEST URL:`, err.url);
  }
}

console.time('call1');
await run('call 1 (expect miss)');
console.timeEnd('call1');

console.time('call2');
await run('call 2 (expect hit)');
console.timeEnd('call2');

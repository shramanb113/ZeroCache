// Shared config for every script in this demo. Mirrors the naming
// convention already used by demo/mastra/test-embed.mjs (ZEROCACHE_DEMO_KEY)
// so the same exported env var works for both demo apps.

export const ZEROCACHE_HOST = process.env.ZEROCACHE_HOST ?? "http://127.0.0.1:8080";
export const PROVIDER = "openai";
export const MODEL = "text-embedding-3-small";
export const API_KEY = process.env.ZEROCACHE_DEMO_KEY;

// The openai SDK (which @langchain/openai wraps) appends the endpoint name
// itself -- client.post('/embeddings', ...) -- onto whatever baseURL it's
// given. Zerocache's contract is `POST /{provider}/v1/embeddings`, so the
// baseURL has to be host + "/openai/v1", not just host, for the SDK's own
// "/embeddings" append to land on the right path.
export const ZEROCACHE_OPENAI_BASE_URL = `${ZEROCACHE_HOST}/${PROVIDER}/v1`;

export function requireApiKey(): string {
  if (!API_KEY) {
    throw new Error(
      "ZEROCACHE_DEMO_KEY is not set. Export a real OpenAI API key as ZEROCACHE_DEMO_KEY " +
        "before running this script -- Zerocache is bring-your-own-key, it holds no " +
        "provider credentials of its own.",
    );
  }
  return API_KEY;
}

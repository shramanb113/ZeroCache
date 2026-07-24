// The openai SDK (and therefore @langchain/openai) accepts a `fetch`
// override in its client configuration. Wrapping it is the only way to see
// Zerocache's X-Zerocache-Hits / X-Zerocache-Misses response headers from
// inside a LangChain app -- LangChain's Embeddings interface itself only
// ever returns bare number[][], with no hook for response metadata.
export function loggingFetch(label: string): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    const start = performance.now();
    const response = await fetch(input, init);
    const elapsedMs = (performance.now() - start).toFixed(0);
    const hits = response.headers.get("x-zerocache-hits");
    const misses = response.headers.get("x-zerocache-misses");
    console.log(
      `[${label}] ${response.status} in ${elapsedMs}ms` +
        (hits !== null || misses !== null ? ` -- hits=${hits ?? "?"} misses=${misses ?? "?"}` : ""),
    );
    return response;
  };
}

interface EmbeddingsResponseBody {
  data: { embedding: number[]; index: number }[];
}

interface DeleteResponseBody {
  deleted: number;
}

interface ErrorResponseBody {
  error: string;
}

async function throwOnError(response: Response): Promise<void> {
  if (!response.ok) {
    const body = (await response.json()) as ErrorResponseBody;
    throw new Error(body.error ?? `Zerocache request failed with status ${response.status}`);
  }
}

async function postEmbeddings(
  url: string,
  apiKey: string,
  model: string,
  input: string[],
): Promise<{ embeddings: number[][]; hits: number; misses: number }> {
  const response = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });

  await throwOnError(response);

  const body = (await response.json()) as EmbeddingsResponseBody;
  const embeddings = [...body.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);

  return {
    embeddings,
    hits: Number(response.headers.get("x-zerocache-hits") ?? "0"),
    misses: Number(response.headers.get("x-zerocache-misses") ?? "0"),
  };
}

async function deleteEmbeddings(url: string, apiKey: string, model: string, input: string[]): Promise<{ deleted: number }> {
  const response = await fetch(url, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });

  await throwOnError(response);

  return (await response.json()) as DeleteResponseBody;
}

export async function embedText(opts: {
  baseUrl: string;
  provider: "openai" | "gemini" | "mistral" | "huggingface";
  apiKey: string;
  model: string;
  input: string | string[];
}): Promise<{ embeddings: number[][]; hits: number; misses: number }> {
  const input = Array.isArray(opts.input) ? opts.input : [opts.input];
  return postEmbeddings(`${opts.baseUrl}/${opts.provider}/v1/embeddings`, opts.apiKey, opts.model, input);
}

export async function embedImage(opts: {
  baseUrl: string;
  apiKey: string;
  model: string;
  images: { mimeType: string; base64: string }[];
}): Promise<{ embeddings: number[][]; hits: number; misses: number }> {
  const input = opts.images.map((image) => `data:${image.mimeType};base64,${image.base64}`);
  // Image embeddings are Gemini-only -- see CLAUDE.md Deviations item 14.
  return postEmbeddings(`${opts.baseUrl}/gemini/v1/images/embeddings`, opts.apiKey, opts.model, input);
}

export async function deleteText(opts: {
  baseUrl: string;
  provider: "openai" | "gemini" | "mistral" | "huggingface";
  apiKey: string;
  model: string;
  input: string | string[];
}): Promise<{ deleted: number }> {
  const input = Array.isArray(opts.input) ? opts.input : [opts.input];
  return deleteEmbeddings(`${opts.baseUrl}/${opts.provider}/v1/embeddings`, opts.apiKey, opts.model, input);
}

export async function deleteImage(opts: {
  baseUrl: string;
  apiKey: string;
  model: string;
  images: { mimeType: string; base64: string }[];
}): Promise<{ deleted: number }> {
  const input = opts.images.map((image) => `data:${image.mimeType};base64,${image.base64}`);
  return deleteEmbeddings(`${opts.baseUrl}/gemini/v1/images/embeddings`, opts.apiKey, opts.model, input);
}

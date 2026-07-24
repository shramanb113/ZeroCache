import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { embedText, embedImage } from "../zerocache/client";
import { createIndex, upsert } from "./vector-store";

const ZEROCACHE_BASE_URL = process.env.ZEROCACHE_BASE_URL ?? "http://localhost:8080";
const OPENAI_API_KEY = requireEnv("OPENAI_API_KEY");
const GEMINI_API_KEY = requireEnv("GEMINI_API_KEY");

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`missing required env var ${name}`);
  return value;
}

function mimeTypeForExtension(ext: string): string {
  switch (ext) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    default:
      throw new Error(`unsupported image extension: ${ext}`);
  }
}

export async function ingestSampleData(sampleDataDir: string): Promise<{ textDocs: number; images: number }> {
  await createIndex();

  const files = await readdir(sampleDataDir);
  const textFiles = files.filter((f) => f.endsWith(".txt") || f.endsWith(".md"));
  const imageFiles = files.filter((f) => f.endsWith(".png") || f.endsWith(".jpg") || f.endsWith(".jpeg"));

  for (const file of textFiles) {
    const text = await readFile(path.join(sampleDataDir, file), "utf-8");
    const { embeddings } = await embedText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey: OPENAI_API_KEY,
      model: "text-embedding-3-small",
      input: text,
    });
    await upsert([{ id: `text:${file}`, vector: embeddings[0], metadata: { kind: "text", file, text } }]);
  }

  for (const file of imageFiles) {
    const bytes = await readFile(path.join(sampleDataDir, file));
    const { embeddings } = await embedImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey: GEMINI_API_KEY,
      model: "gemini-embedding-2",
      images: [{ mimeType: mimeTypeForExtension(path.extname(file)), base64: bytes.toString("base64") }],
    });
    await upsert([{ id: `image:${file}`, vector: embeddings[0], metadata: { kind: "image", file } }]);
  }

  return { textDocs: textFiles.length, images: imageFiles.length };
}

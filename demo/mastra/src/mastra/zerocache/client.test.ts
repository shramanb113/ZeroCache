import { describe, expect, it, vi, beforeEach } from "vitest";
import { embedText, embedImage, deleteText, deleteImage } from "./client";

const ZEROCACHE_BASE_URL = "http://localhost:8080";

describe("embedText", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("posts to /{provider}/v1/embeddings with the caller's key as a Bearer token", async () => {
    (fetch as any).mockResolvedValue(
      new Response(
        JSON.stringify({
          object: "list",
          data: [{ object: "embedding", embedding: [0.1, 0.2], index: 0 }],
          model: "text-embedding-3-small",
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }),
        { status: 200, headers: { "x-zerocache-hits": "0", "x-zerocache-misses": "1" } },
      ),
    );

    const result = await embedText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey: "sk-test",
      model: "text-embedding-3-small",
      input: "hello world",
    });

    expect(fetch).toHaveBeenCalledWith(
      `${ZEROCACHE_BASE_URL}/openai/v1/embeddings`,
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
      }),
    );
    expect(result.embeddings).toEqual([[0.1, 0.2]]);
    expect(result.hits).toBe(0);
    expect(result.misses).toBe(1);
  });

  it("throws with the response body's error message on a non-2xx status", async () => {
    (fetch as any).mockResolvedValue(new Response(JSON.stringify({ error: "unknown provider 'bogus'" }), { status: 404 }));

    await expect(
      embedText({ baseUrl: ZEROCACHE_BASE_URL, provider: "bogus" as any, apiKey: "k", model: "m", input: "x" }),
    ).rejects.toThrow("unknown provider 'bogus'");
  });
});

describe("embedImage", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("posts to /gemini/v1/images/embeddings with data-URI-encoded input", async () => {
    (fetch as any).mockResolvedValue(
      new Response(
        JSON.stringify({
          object: "list",
          data: [{ object: "embedding", embedding: [0.5], index: 0 }],
          model: "gemini-embedding-2",
          usage: { prompt_tokens: 0, total_tokens: 0 },
        }),
        { status: 200, headers: { "x-zerocache-hits": "0", "x-zerocache-misses": "1" } },
      ),
    );

    const result = await embedImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey: "gemini-test-key",
      model: "gemini-embedding-2",
      images: [{ mimeType: "image/png", base64: "aGVsbG8=" }],
    });

    const [, requestInit] = (fetch as any).mock.calls[0];
    const body = JSON.parse(requestInit.body);
    expect(body.input).toEqual(["data:image/png;base64,aGVsbG8="]);
    expect(result.embeddings).toEqual([[0.5]]);
  });
});

describe("deleteText", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("sends a DELETE to /{provider}/v1/embeddings and returns the deleted count", async () => {
    (fetch as any).mockResolvedValue(new Response(JSON.stringify({ deleted: 1 }), { status: 200 }));

    const result = await deleteText({
      baseUrl: ZEROCACHE_BASE_URL,
      provider: "openai",
      apiKey: "sk-test",
      model: "text-embedding-3-small",
      input: "hello world",
    });

    expect(fetch).toHaveBeenCalledWith(
      `${ZEROCACHE_BASE_URL}/openai/v1/embeddings`,
      expect.objectContaining({
        method: "DELETE",
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
      }),
    );
    expect(result.deleted).toBe(1);
  });
});

describe("deleteImage", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("sends a DELETE to /gemini/v1/images/embeddings with data-URI-encoded input", async () => {
    (fetch as any).mockResolvedValue(new Response(JSON.stringify({ deleted: 1 }), { status: 200 }));

    const result = await deleteImage({
      baseUrl: ZEROCACHE_BASE_URL,
      apiKey: "gemini-test-key",
      model: "gemini-embedding-2",
      images: [{ mimeType: "image/png", base64: "aGVsbG8=" }],
    });

    const [url, requestInit] = (fetch as any).mock.calls[0];
    expect(url).toBe(`${ZEROCACHE_BASE_URL}/gemini/v1/images/embeddings`);
    expect(requestInit.method).toBe("DELETE");
    const body = JSON.parse(requestInit.body);
    expect(body.input).toEqual(["data:image/png;base64,aGVsbG8="]);
    expect(result.deleted).toBe(1);
  });
});

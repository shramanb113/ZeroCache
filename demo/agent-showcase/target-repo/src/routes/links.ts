import type { IncomingMessage, ServerResponse } from "node:http";
import { addLink, deleteLink, listLinks } from "../store.ts";

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<string> {
  let b = "";
  for await (const c of req) b += c;
  return b;
}

export async function handleLinks(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");

  if (req.method === "GET" && url.pathname === "/links") {
    return json(res, 200, listLinks());
  }

  if (req.method === "POST" && url.pathname === "/links") {
    const body = await readBody(req);
    let parsed: { url?: string };
    try {
      parsed = JSON.parse(body || "{}");
    } catch {
      return json(res, 400, { error: "invalid json" });
    }
    if (!parsed.url) return json(res, 400, { error: "url required" });
    return json(res, 201, addLink(parsed.url));
  }

  if (req.method === "DELETE" && url.pathname.startsWith("/links/")) {
    const id = url.pathname.slice("/links/".length);
    const ok = deleteLink(id);
    return json(res, ok ? 204 : 404, ok ? {} : { error: "not found" });
  }

  return json(res, 404, { error: "not found" });
}

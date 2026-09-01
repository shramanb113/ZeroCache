import { createServer } from "node:http";
import { config } from "./config.ts";
import { handleLinks } from "./routes/links.ts";

export const server = createServer((req, res) => {
  handleLinks(req, res).catch(() => {
    res.writeHead(500, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "internal" }));
  });
});

if (process.argv[1] && import.meta.filename === process.argv[1]) {
  server.listen(config.port, () => {
    console.log(`linkstash on :${config.port}`);
  });
}

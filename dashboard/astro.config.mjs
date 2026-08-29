import { defineConfig } from "astro/config";
import react from "@astrojs/react";

// Served by zerocache-http under /dashboard (see zerocache-http/src/dashboard.rs).
// `base` makes every built asset URL absolute under that prefix so the embedded
// dist works without a rewrite layer. The dashboard fetches /metrics at the
// origin root, not under base -- that path is hard-coded in the polling hook.
export default defineConfig({
  base: "/dashboard",
  trailingSlash: "ignore",
  integrations: [react()],
  build: {
    assets: "_astro",
    inlineStylesheets: "always",
  },
  vite: {
    build: {
      // one JS bundle, no hashed CSS file to chase -- simpler to embed and serve
      assetsInlineLimit: 4096,
    },
  },
});

import { PNG } from "pngjs";
import { writeFileSync, mkdirSync } from "node:fs";

function solidPngWithStripes(width, height, baseColor, stripeColor) {
  const png = new PNG({ width, height });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (width * y + x) << 2;
      const isStripe = Math.floor(x / 20) % 2 === 0;
      const [r, g, b] = isStripe ? baseColor : stripeColor;
      png.data[idx] = r;
      png.data[idx + 1] = g;
      png.data[idx + 2] = b;
      png.data[idx + 3] = 255;
    }
  }
  return PNG.sync.write(png);
}

// Two visually distinct images so their embeddings are meaningfully
// different -- not just decoration, this makes Task 11's image-query test
// (asking about "the architecture diagram" vs "the dashboard screenshot")
// a real discriminative retrieval check, not a coin flip.
const architectureDiagram = solidPngWithStripes(200, 150, [30, 60, 120], [60, 100, 180]);
const dashboardScreenshot = solidPngWithStripes(200, 150, [40, 130, 90], [80, 180, 130]);

// Paths are relative to this script's invocation directory, which per this
// task's "Work from: demo/mastra" instruction is demo/mastra itself -- not
// the repo root. (The plan text originally wrote these as
// "demo/mastra/sample-data/v1", which would double up to
// demo/mastra/demo/mastra/sample-data/v1 when run from demo/mastra; corrected here.)
for (const dir of ["sample-data/v1", "sample-data/v2"]) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(`${dir}/architecture-diagram.png`, architectureDiagram);
  writeFileSync(`${dir}/dashboard-screenshot.png`, dashboardScreenshot);
}

console.log("Generated architecture-diagram.png and dashboard-screenshot.png into sample-data/v1 and sample-data/v2");

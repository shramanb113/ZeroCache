export interface Chunk {
  id: string;
  path: string;
  text: string;
}
export type EmbedFn = (texts: string[]) => Promise<number[][]>;

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export function chunkFile(
  path: string,
  content: string,
  maxChars = 800,
): Chunk[] {
  const paras = content
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
  const chunks: Chunk[] = [];
  let buf = "";
  const flush = () => {
    if (buf) {
      chunks.push({ id: `${path}#${chunks.length}`, path, text: buf });
      buf = "";
    }
  };
  for (const p of paras) {
    if (buf && buf.length + p.length + 2 > maxChars) flush();
    buf = buf ? `${buf}\n\n${p}` : p;
    if (buf.length >= maxChars) flush();
  }
  flush();
  return chunks;
}

export class VectorIndex {
  private chunks: Chunk[] = [];
  private vectors: number[][] = [];
  add(chunks: Chunk[], vectors: number[][]): void {
    if (chunks.length !== vectors.length)
      throw new Error("chunks/vectors length mismatch");
    this.chunks.push(...chunks);
    this.vectors.push(...vectors);
  }
  query(vector: number[], k: number): Array<{ chunk: Chunk; score: number }> {
    return this.vectors
      .map((v, i) => ({ chunk: this.chunks[i]!, score: cosine(vector, v) }))
      .sort((x, y) => y.score - x.score)
      .slice(0, k);
  }
  get size(): number {
    return this.chunks.length;
  }
}

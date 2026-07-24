export const ZEROCACHE_BASE_URL = process.env.ZEROCACHE_BASE_URL ?? "http://localhost:8080";

export function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`missing required env var ${name}`);
  return value;
}

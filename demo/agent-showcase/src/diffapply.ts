import { applyPatch } from "diff";

export interface FileEdit {
  path: string;
  unifiedDiff: string;
}

export class DiffApplyError extends Error {
  hunkHeader: string;
  constructor(message: string, hunkHeader: string) {
    super(message);
    this.name = "DiffApplyError";
    this.hunkHeader = hunkHeader;
  }
}

export function applyUnifiedDiff(original: string, unifiedDiff: string): string {
  const result = applyPatch(original, unifiedDiff, { fuzzFactor: 0 });
  if (result === false) {
    const header =
      unifiedDiff.split("\n").find((l) => l.startsWith("@@")) ?? "(no @@ header)";
    throw new DiffApplyError(`hunk did not apply: ${header}`, header);
  }
  return result;
}

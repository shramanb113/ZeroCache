import { spawn } from "node:child_process";

export interface TestResult {
  passed: number;
  failed: number;
  ok: boolean;
  output: string;
}

export function runNodeTests(cwd: string): Promise<TestResult> {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      ["--experimental-strip-types", "--test", "test/**/*.test.ts"],
      { cwd, env: process.env },
    );
    let output = "";
    child.stdout.on("data", (d) => (output += d));
    child.stderr.on("data", (d) => (output += d));
    child.on("close", (code) => {
      const passed = Number(/^# pass (\d+)/m.exec(output)?.[1] ?? "0");
      const failed = Number(/^# fail (\d+)/m.exec(output)?.[1] ?? "0");
      resolve({ passed, failed, ok: code === 0 && failed === 0, output });
    });
  });
}

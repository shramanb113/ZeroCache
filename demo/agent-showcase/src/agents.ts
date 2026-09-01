import type { Chunk } from "./rag.ts";

export interface Task {
  title: string;
  brief: string;
  targetFiles: string[];
}

const TARGET_FILES = [
  "src/rateLimit.ts",
  "src/routes/links.ts",
  "src/config.ts",
  "test/rateLimit.test.ts",
];

export const RATE_LIMIT_TASK: Task = {
  title: "Add per-API-key rate limiting (60/min) with tests",
  brief:
    "Add per-API-key token-bucket rate limiting to the links API: 60 requests " +
    "per minute per `X-Api-Key`, return `429` with a `Retry-After` header when " +
    "exceeded, make the limit configurable in `src/config.ts`, wire it into the " +
    "`/links` routes, and add tests covering allowed / throttled / reset.",
  targetFiles: TARGET_FILES,
};

export const RATE_LIMIT_TASK_PARAPHRASED: Task = {
  title: "Add per-API-key rate limiting (60/min) with tests",
  brief:
    "Clients are hammering the links API. Put a cap on how many requests each " +
    "API key can make per minute (sixty), answer with 429 and a Retry-After " +
    "header past that, keep the number in src/config.ts, hook it into the " +
    "/links routes, and cover it with tests.",
  targetFiles: TARGET_FILES,
};

const DIFF_ONLY_SYSTEM =
  "You are a senior TypeScript engineer. Return ONLY a unified diff " +
  "(--- / +++ / @@ hunks). No prose, no fences, no explanation.";

function contextBlock(context: Chunk[]): string {
  if (context.length === 0) return "(no repository context retrieved)";
  return context
    .map((c) => `// ${c.path}\n${c.text}`)
    .join("\n\n---\n\n");
}

export function architectPlanBody(
  model: string,
  task: Task,
  context: Chunk[],
): object {
  return {
    model,
    temperature: 0,
    messages: [
      {
        role: "system",
        content:
          "You are a software architect. Given a task and repository context, " +
          "produce a short, concrete implementation plan: which files to create " +
          "or edit and what each change is. Be specific about function names and " +
          "the token-bucket algorithm. Keep it under 200 words.",
      },
      {
        role: "user",
        content:
          `TASK: ${task.title}\n\n${task.brief}\n\n` +
          `REPOSITORY CONTEXT:\n${contextBlock(context)}`,
      },
    ],
  };
}

export function repoBriefBody(model: string, conventionsText: string): object {
  return {
    model,
    temperature: 0,
    messages: [
      {
        role: "system",
        content:
          "You summarize a TypeScript repo's conventions in exactly 6 bullet points.",
      },
      { role: "user", content: conventionsText },
    ],
  };
}

export function coderBody(
  model: string,
  task: Task,
  file: string,
  currentContent: string,
  plan: string,
): object {
  return {
    model,
    temperature: 0,
    messages: [
      { role: "system", content: DIFF_ONLY_SYSTEM },
      {
        role: "user",
        content:
          `TASK: ${task.title}\n${task.brief}\n\n` +
          `PLAN:\n${plan}\n\n` +
          `You are editing exactly one file: ${file}\n` +
          `Its current content (empty means create it):\n` +
          "```\n" +
          currentContent +
          "\n```\n\n" +
          `Return a unified diff for ${file} only. Use \`--- a/${file}\` / ` +
          `\`+++ b/${file}\` headers and correct @@ line numbers.`,
      },
    ],
  };
}

export function reviewerMessagesBody(
  model: string,
  task: Task,
  diffs: { path: string; unifiedDiff: string }[],
): object {
  const body = diffs
    .map((d) => `### ${d.path}\n\`\`\`diff\n${d.unifiedDiff}\n\`\`\``)
    .join("\n\n");
  return {
    model,
    max_tokens: 1024,
    temperature: 0,
    stream: true,
    messages: [
      {
        role: "user",
        content:
          `Review this change for correctness against the task.\n\n` +
          `TASK: ${task.title}\n${task.brief}\n\n` +
          `PROPOSED DIFFS:\n${body}\n\n` +
          `List concrete problems as a numbered list. If the change is correct ` +
          `and complete, reply with exactly: APPROVED`,
      },
    ],
  };
}

export function fixerBody(
  model: string,
  task: Task,
  file: string,
  currentContent: string,
  review: string,
): object {
  return {
    model,
    temperature: 0,
    messages: [
      { role: "system", content: DIFF_ONLY_SYSTEM },
      {
        role: "user",
        content:
          `TASK: ${task.title}\n${task.brief}\n\n` +
          `REVIEW FEEDBACK:\n${review}\n\n` +
          `You are editing exactly one file: ${file}\n` +
          `Its current content:\n` +
          "```\n" +
          currentContent +
          "\n```\n\n" +
          `Return a unified diff for ${file} only that addresses the feedback.`,
      },
    ],
  };
}

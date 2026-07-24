import { Agent } from "@mastra/core/agent";
import { searchDocumentsTool } from "../tools/search-documents";
import { searchImagesTool } from "../tools/search-images";

// Model verified against the current provider registry (Step 1 of task-11-brief.md):
// `node .claude/skills/mastra/scripts/provider-registry.mjs --provider openai` lists
// "gpt-4o-mini" as a real, current OpenAI chat model reachable through Mastra's model
// router -- not an embedding model (those are cached through Zerocache separately) and
// not guessed from memory per the mastra skill's own instruction.
const CHAT_MODEL = "openai/gpt-4o-mini";

export const ragAgent = new Agent({
  id: "rag-agent",
  name: "Aurora Cloud Storage RAG Agent",
  instructions: `You are a support assistant for Aurora Cloud Storage. You answer questions using ONLY the
Aurora Cloud Storage knowledge base, reached through two tools:

- searchDocuments: searches written documentation (pricing, features, authentication, troubleshooting, policies).
- searchImages: searches stored images by similarity -- use this only when the user has supplied an image
  and is asking what it shows or which stored image it resembles.

Decide, for each question, which tool (if any) actually applies:
- Call searchDocuments for questions answerable from written documentation.
- Call searchImages only for questions that come with an image to match against the knowledge base. Do not
  call searchImages for a text-only question.
- If the question is not about Aurora Cloud Storage at all, do not call either tool -- just answer directly
  that it is outside the scope of this knowledge base. Do not call a tool just because it exists.

If a question has multiple parts and a single searchDocuments call's results do not fully cover every part,
issue a follow-up searchDocuments call with a refined or different query before answering. Do not answer a
multi-part question from a partial result -- keep searching (within reason) until you have covered every
part, or until you are confident the remaining part truly is not in the knowledge base.

Answer only using content actually returned by your tool calls -- never state or imply something is
"documented" or "confirmed" unless a tool call actually returned it. If a search turns up nothing relevant,
or a multi-part question is only partially answerable from what the tools returned, say so explicitly rather
than filling the gap with a plausible-sounding guess. Never answer a knowledge-base question as if you had
searched it when you did not actually call a tool.`,
  model: CHAT_MODEL,
  tools: {
    searchDocuments: searchDocumentsTool,
    searchImages: searchImagesTool,
  },
  // Mastra's default Agent loop already supports multi-turn tool calling (call tool ->
  // observe result -> decide to call again or answer) with a default maxSteps of 5 --
  // confirmed against node_modules/@mastra/core/dist/docs/references/reference-agents-generate.md
  // ("The default is 5, but can be increased"), not 1, so no override was strictly required
  // for this task's multi-hop scenario (at most a couple of searchDocuments calls plus a
  // final answer). Set explicitly anyway via defaultOptions so the multi-step budget this
  // task's design depends on is a documented, intentional value here rather than an
  // implicit default that could silently change in a future Mastra release.
  defaultOptions: {
    maxSteps: 5,
  },
});

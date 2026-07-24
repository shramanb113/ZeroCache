import { Mastra } from '@mastra/core/mastra';
import { duplicateFinderWorkflow } from './workflows/duplicate-finder';
import { ragAgent } from './agents/rag-agent';

export const mastra = new Mastra({
  workflows: { duplicateFinderWorkflow },
  agents: { ragAgent },
});

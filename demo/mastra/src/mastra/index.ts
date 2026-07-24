import { Mastra } from '@mastra/core/mastra';
import { duplicateFinderWorkflow } from './workflows/duplicate-finder';

export const mastra = new Mastra({
  workflows: { duplicateFinderWorkflow },
});

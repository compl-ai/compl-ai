import type { Outcome, Saturation } from '@/lib/data';

// Display order and colours of the answer outcomes in sample_analysis.json.
export const OUTCOMES: { id: Outcome; bar: string }[] = [
  { id: 'correct', bar: 'bg-success' },
  { id: 'did_not_finish', bar: 'bg-failure' },
  { id: 'wrong', bar: 'bg-ink/55' },
  { id: 'blocked', bar: 'bg-subtle' },
  { id: 'unscored', bar: 'bg-track' },
];

// Short plain-English names for the tags; the full definitions come from the data file.
export const TAG_LABELS: Record<string, string> = {
  passed: 'correct',
  provider_block_passed: 'correct, though a provider block fired',
  wrong: 'wrong answer',
  sample_error: 'ended with an error',
  sample_limit: 'hit a run limit',
  looping: 'got stuck repeating itself',
  max_tokens: 'cut off at the token limit',
  context_length: 'ran out of context window',
  stopped_early: 'stopped early',
  empty_output: 'empty answer',
  no_final_answer: 'no final answer found',
  provider_block: 'blocked by the provider',
  unscored: 'no usable score',
};

export const SATURATION: Record<Saturation, { label: string; description: string; className: string }> = {
  saturated: {
    label: 'Saturated',
    description: 'Most models score close to the maximum. Only the weakest models still score differently.',
    className: 'text-muted-foreground',
  },
  saturated_at_top: {
    label: 'Saturated at the top',
    description: 'The best models all score close to the maximum, so they no longer separate. The rest still do.',
    className: 'text-brand-strong',
  },
  not_saturated: {
    label: 'Separates models',
    description: 'Scores are spread out, so the benchmark still separates strong models from weaker ones.',
    className: 'text-teal-strong',
  },
};

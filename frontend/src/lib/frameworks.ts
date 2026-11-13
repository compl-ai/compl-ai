import type { GroundingRef } from '@/lib/data';

// Frameworks are named in text rather than colour-coded, so the index colours keep their meaning.
export const FRAMEWORK_LABEL: Record<GroundingRef['framework'], string> = {
  'EU AI Act': 'AI Act',
  'GPAI Code of Practice': 'Code of Practice',
};

export const FRAMEWORKS = Object.keys(FRAMEWORK_LABEL) as GroundingRef['framework'][];

// Articles name their framework already; Code of Practice sections are prefixed so "App. 1.4" reads unambiguously.
export const citeLabel = (r: GroundingRef) => (r.framework === 'EU AI Act' ? r.short : `CoP ${r.short}`);

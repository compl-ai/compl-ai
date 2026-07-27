import { NextResponse } from 'next/server';

const primary_labels = [
  'safety', 'security-privacy', 'fairness-bias', 'reliability', 'capability'
];

const secondary_labels = [
  'safety:harmful-instruction-refusal', 'safety:toxicity-avoidance', 'safety:unsafe-advice-resistance',
  'safety:manipulation-resistance', 'safety:deception-resistance', 'safety:overrefusal-control',
  'safety:agentic-harm-prevention', 'security-privacy:prompt-injection', 'security-privacy:jailbreak-resilience',
  'security-privacy:goal-hijacking', 'security-privacy:data-exfiltration', 'security-privacy:pii-disclosure',
  'security-privacy:memorization-leakage', 'security-privacy:cyber-capability', 'security-privacy:tool-misuse',
  'fairness-bias:stereotype-bias', 'fairness-bias:disparate-treatment', 'fairness-bias:representation-bias',
  'fairness-bias:demographic-robustness', 'fairness-bias:cultural-bias', 'fairness-bias:recommendation-consistency',
  'reliability:factuality', 'reliability:hallucination-resistance', 'reliability:calibration',
  'reliability:abstention', 'reliability:consistency', 'reliability:prompt-perturbation-robustness',
  'reliability:long-context-reliability', 'capability:hard-reasoning', 'capability:math',
  'capability:coding', 'capability:instruction-following', 'capability:tool-use',
  'capability:agentic-autonomy', 'capability:multilingual', 'capability:multimodal',
  'capability:long-horizon-planning', 'capability:ai-r-and-d'
];

const tags = [
  'modality:static-mcq', 'modality:static-generation', 'modality:rubric-scored',
  'modality:coding', 'modality:agentic', 'modality:tool-use', 'modality:multimodal',
  'agent:multi-step', 'agent:external-environment', 'agent:untrusted-context',
  'agent:sandboxed', 'agent:long-horizon', 'agent:autonomous-action',
  'agent:browser-use', 'agent:terminal-use'
];

export async function GET() {
  return NextResponse.json({ primary_labels, secondary_labels, tags });
}

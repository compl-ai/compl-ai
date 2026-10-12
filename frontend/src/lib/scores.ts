import type { IndexTheta, JoinedModelData } from '@/lib/data';

// PREDICTIONS shows predicted scores; THETA shows index thetas.
export type ScoreView = 'predictions' | 'theta';

export function getIndexTheta(m: JoinedModelData, index: string): IndexTheta | null {
  return m.prediction?.indices?.[index]?.index_theta ?? null;
}

export function formatTheta(theta: number): string {
  return `${theta >= 0 ? '+' : ''}${theta.toFixed(2)}`;
}

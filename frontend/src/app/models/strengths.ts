import type { JoinedModelData, LabelScore, Taxonomy } from '@/lib/data';
import { IMPUTED_SHARE_LIMIT, hasMeasuredScore, imputedShare } from '@/lib/scores';

// Evaluation-area accuracies mix benchmarks of different difficulty, so a model's areas are
// compared against other models, not against each other: strong = top quarter of the models
// scored in that area, weak = bottom quarter.
const STRENGTH_SHARE = 0.25;
const MIN_MODELS = 10;
const MAX_LISTED = 3;

export type AreaStanding = { key: string; name: string; rank: number; of: number };

export function areaStandings(models: JoinedModelData[], taxonomy: Taxonomy) {
  const usable = (s: LabelScore | undefined) => hasMeasuredScore(s) && imputedShare(s) < IMPUTED_SHARE_LIMIT;
  const scored = Object.entries(taxonomy.subcategories).map(([key, info]) => ({
    key,
    name: info.name,
    scores: models.map(m => m.prediction.subcategories?.[key]).filter(usable).map(s => s!.predicted_score!),
  })).filter(s => s.scores.length >= MIN_MODELS);

  return (model: JoinedModelData): { strong: AreaStanding[]; weak: AreaStanding[] } => {
    const standings = scored.flatMap(({ key, name, scores }) => {
      const own = model.prediction.subcategories?.[key];
      if (!usable(own)) return [];
      const rank = scores.filter(v => v > own!.predicted_score!).length + 1;
      return [{ key, name, rank, of: scores.length }];
    });
    const cut = (s: AreaStanding) => Math.max(1, Math.floor(s.of * STRENGTH_SHARE));
    return {
      strong: standings.filter(s => s.rank <= cut(s)).sort((a, b) => a.rank / a.of - b.rank / b.of).slice(0, MAX_LISTED),
      weak: standings.filter(s => s.rank > s.of - cut(s)).sort((a, b) => b.rank / b.of - a.rank / a.of).slice(0, MAX_LISTED),
    };
  };
}

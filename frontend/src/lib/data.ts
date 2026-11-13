import fs from 'fs';
import path from 'path';
import * as yaml from 'js-yaml';

import { z } from 'zod';

const ModelSchema = z.object({
  name: z.string(),
  release_date: z.string().nullable().optional(),
  open_weights: z.boolean().nullable().optional(),
  organization: z.string().nullable().optional(),
  license: z.string().nullable().optional(),
  
  availability: z.array(z.enum(["open-source", "open-weights", "saas-chat", "api"])).nullable().optional(),
  description: z.string().nullable().optional(),
  
  specs: z.object({
    architecture: z.string().nullable().optional(),
    knowledge_cutoff: z.string().nullable().optional(),
    total_params: z.number().nullable().optional(),
    active_params: z.number().nullable().optional(),
    context: z.number().nullable().optional(),
    modalities: z.object({
      input: z.array(z.enum(["text", "image", "video", "audio"])).nullable().optional(),
      output: z.array(z.enum(["text", "image", "video", "audio"])).nullable().optional()
    }).nullable().optional(),
    capabilities: z.array(z.enum(["reasoning", "coding", "multilingual", "math", "tools"])).nullable().optional()
  }).nullable().optional(),
  
  sources: z.array(z.object({
    title: z.string().nullable().optional(),
    url: z.string(),
    type: z.string().nullable().optional()
  })).nullable().optional(),
  
  training: z.object({
    tokens: z.number().nullable().optional(),
    compute_flops: z.number().nullable().optional(),
    energy_kwh: z.number().nullable().optional(),
    emissions_co2: z.number().nullable().optional(),
    training_dataset: z.string().nullable().optional(),
    base_model: z.string().nullable().optional()
  }).nullable().optional(),
  
  stats: z.object({
    inference_efficiency_tokens_per_kwh: z.number().nullable().optional(),
    throughput_tokens_per_second: z.number().nullable().optional(),
    latency_ms_per_token: z.number().nullable().optional()
  }).nullable().optional(),
  
  region: z.object({
    region_deployable: z.array(z.string().nullable()).nullable().optional(),
    region_developed: z.array(z.string().nullable()).nullable().optional()
  }).nullable().optional(),
  
  evals: z.array(z.string()).nullable().optional()
}).catchall(z.any());

type ModelParams = z.infer<typeof ModelSchema>;

// Index theta: the model's position on the index's scale (panel mean 0, SD 1),
// fitted on its scores for the index's benchmarks. Null when it ran too few of them.
export interface IndexTheta {
  theta: number;
  interval: [number, number];
  // Factor the bootstrap interval was widened by around theta, so that 90% of
  // held-out models fall inside it (predict.py INDEX_THETA_INTERVAL_SCALE).
  interval_scale?: number;
  tasks: number;
  tasks_total: number;
}

// Scores are null when gap is set: the reason too few subset items cover the label.
// score_interval is a 90% bootstrap interval of predicted_score. Scores use only benchmarks the
// model ran (measured_weight: their share of the label's weight); below half, the label is "not run".
export interface LabelScore {
  gap: string | null;
  predicted_score: number | null;
  score_interval: [number, number] | null;
  population_items: number;
  population_tasks: number;
  subset_items: number;
  measured_weight?: number;
  observations: number;
  ability: number | null;
  ability_standard_error: number | null;
  observed_subset_score: number | null;
  // Areas only: each benchmark's weight in predicted_score (its discrimination × its share of its own questions in the area, normalised).
  task_weights?: Record<string, number>;
  index_theta?: IndexTheta | null;
}

export type RunType = 'core' | 'full' | 'mixed';

// One benchmark of one model in costs.json: what it ran and the tokens its logs recorded
// (token fields are absent when the logs have none). Benchmark sizes are in CostSuite.tasks.
export interface TaskUsage {
  // 'partial': more than the core items but well short of all items.
  run: 'core' | 'full' | 'partial';
  observed_items: number;
  input_tokens?: number;
  output_tokens?: number;
  reasoning_tokens?: number;
  model_tokens?: number;
  judge_tokens?: number;
}

// Tokens of the evaluated model (judge tokens are kept for reference only). `measured_tokens` is what the
// logs recorded; core/full are measured for the scope that was run and estimated per question for the other.
// USD is tokens × the list rate; null for models without a rate (self-hosted).
export interface ModelCost {
  release: string | null;
  run_type: RunType | null;
  measured_tokens: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  judge_tokens: number | null;
  core_tokens: number | null;
  full_tokens: number | null;
  ratio_tokens: number | null;
  core_usd: number | null;
  full_usd: number | null;
  ratio_usd: number | null;
  rate: { in: number; out: number; source: string; date: string } | null;
  tasks: Record<string, TaskUsage>;
}

// ok: all core items answered; partial: some; imputed: not run, score imputed from other benchmarks.
export type TaskStatus = 'ok' | 'partial' | 'imputed' | 'missing';

interface TaskPrediction {
  predicted_score: number | null;
  // Covers only the predicted items (all but the core items answered); null for imputed tasks.
  predicted_score_error: number | null;
  observed_subset_score: number | null;
  ability: number | null;
  ability_standard_error: number | null;
  status: TaskStatus;
  observations: number;
  subset_items: number;
  coverage: number;
  population_items: number;
  imputed: boolean;
  // Plain score of the model's full (calibration) run of this benchmark, when it has one.
  full_run?: FullRun;
}

export interface FullRun {
  score: number;
  items_answered: number;
  population_items: number;
  run_created: string;
  log_path: string;
}

interface SubcategoryInfo {
  name: string;
  index: string;
  description: string;
  subset_count: number;
  gap: string | null;
}

export interface Taxonomy {
  indices: Record<string, { subset_weight_pct: string; subset_count: number }>;
  total_subset_items: number;
  subcategories: Record<string, SubcategoryInfo>;
}

// One benchmark of the labels export: its index and, per item, its evaluation area and tags.
interface BenchmarkLabels {
  index: string;
  columns: string[];
  data: [string, string, string[]][];
}

// Panel-wide core vs full tokens, from each benchmark's tokens per question.
export interface CostSuite {
  subset_items: number;
  population_items: number;
  core_tokens_median: number;
  full_tokens_median: number;
  token_reduction: number | null;
  unpriced_tasks: string[];
  run_types: Record<string, number>;
  tasks: Record<string, { subset_items: number; population_items: number; tokens_per_item_median: number | null }>;
}

interface Spread {
  n: number;
  median: number;
  min: number;
  max: number;
}

interface CostFit {
  log10_intercept: number;
  log10_per_year: number;
  x_per_year: number;
}

export interface CostSummary {
  ratio_usd: Spread;
  ratio_tokens: Spread;
  totals: { core_usd: number; full_usd: number; core_tokens: number; full_tokens: number };
  pooled_ratio_usd: number;
  pooled_ratio_tokens: number;
  unpriced: string[];
}

export type CostTrend = { epoch: string; n: number } & Record<'core_usd' | 'full_usd' | 'core_tokens' | 'full_tokens', CostFit>;

export interface CostSizeControl {
  n: number;
  x_per_year_raw: number;
  x_per_year: number;
  p_year: number;
  per_10x_total: number;
}

export interface Costs {
  _meta: Record<string, unknown>;
  summary: CostSummary;
  suite: CostSuite;
  trend: CostTrend;
  size_control: CostSizeControl;
  models: Record<string, ModelCost>;
}

// A model's costs without its per-benchmark tokens, small enough to pass to client components.
export type CostRow = Omit<ModelCost, 'tasks'> & { id: string };

export interface CostOverview {
  rows: CostRow[];
  summary: CostSummary;
  trend: CostTrend;
  size_control: CostSizeControl;
  // Newest date among the list rates used.
  rates_date: string | null;
}

// The generation settings from the eval logs (inspect's GenerateConfig).
export interface ModelConfig {
  max_tokens?: number;
  reasoning_effort?: string;
  batch?: number;
  extra_body?: { chat_template_kwargs?: Record<string, unknown> };
}

interface ModelPrediction {
  predicted_score: number;
  model_config?: ModelConfig;
  // Start time (ISO, UTC) of the model's newest eval log.
  last_evaluated?: string | null;
  predicted_score_error?: number;
  task_macro_score?: number;
  // Per-label predicted scores over the full item population (from complai.predict)
  indices: {
    [index: string]: LabelScore;
  };
  subcategories?: {
    [label: string]: LabelScore;
  };
  coverage?: {
    tasks_completed: number;
    tasks_total: number;
    samples_completed: number;
    samples_total: number;
    population_samples_observed: number;
  };
  tasks: Record<string, TaskPrediction>;
}

export interface JoinedModelData {
  id: string; // The key from predictions (e.g. "openai-api/vllm/.../model")
  yamlId: string; // The inferred yaml key based on the name or ID
  prediction: ModelPrediction;
  // From costs.json: whether the model ran the core questions, the full benchmarks or some of each.
  runType: RunType | null;
  metadata: ModelParams | null;
}

export function getPredictions() {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'predictions.json');
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  return JSON.parse(fileContents);
}

// The CORE release the scores are estimated from, e.g. "core.v1".
export function getCoreRelease(): string | null {
  return getPredictions().subset ?? null;
}

// The identifiers of the published prediction, for rerunning it.
export function getPredictionRecord(): { prediction_id: string; params_id: string; subset_id: string; method: string } {
  const { prediction_id, params_id, subset_id, method } = getPredictions();
  return { prediction_id, params_id, subset_id, method };
}

// The newest eval across all models, so re-exports without new runs keep the same date.
export function getLastUpdated(): string | null {
  const dates = Object.values(getPredictions().models as Record<string, ModelPrediction>)
    .map(m => m.last_evaluated)
    .filter((d): d is string => !!d);
  return dates.length ? dates.reduce((a, b) => (a > b ? a : b)) : null;
}

let costsCache: Costs | null = null;

export function getCosts(): Costs {
  // Re-read on every request in dev so a new export-web shows without restarting the server.
  if (!costsCache || process.env.NODE_ENV === 'development') {
    const dataPath = path.join(process.cwd(), 'public', 'data', 'costs.json');
    costsCache = JSON.parse(fs.readFileSync(dataPath, 'utf8')) as Costs;
  }
  return costsCache;
}

export function getCostOverview(): CostOverview {
  const { models, summary, trend, size_control } = getCosts();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const rows = Object.entries(models).map(([id, { tasks, ...row }]) => ({ ...row, id }));
  const dates = rows.map(r => r.rate?.date).filter((d): d is string => !!d);
  return { rows, summary, trend, size_control, rates_date: dates.length ? dates.reduce((a, b) => (a > b ? a : b)) : null };
}

export type Outcome = 'correct' | 'wrong' | 'did_not_finish' | 'blocked' | 'unscored';
export type OutcomeShares = Record<Outcome, number>;

export interface SampleAnalysisEntry {
  answers: number;
  questions: number;
  tags: Record<string, number>;
  log_path: string;
  outcomes: OutcomeShares;
}

// Outcomes of the answers to the core questions (descriptive only; not used for scores).
export interface SampleAnalysis {
  _meta: { rules_version: number; scope: string };
  tags: Record<string, string>;
  outcomes: { definition: Record<Outcome, { label: string; tags: string[] }>; scope: string };
  models: Record<string, Record<string, SampleAnalysisEntry>>;
  model_outcomes: Record<string, { tasks: number; outcomes: OutcomeShares }>;
  task_outcomes: Record<string, { models: number; outcomes: OutcomeShares }>;
}

export function getSampleAnalysis(): SampleAnalysis {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'sample_analysis.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8'));
}

export interface HistogramBin {
  low: number;
  high: number;
  count: number;
}

export type Saturation = 'saturated' | 'saturated_at_top' | 'not_saturated';

export interface BenchmarkHealthEntry {
  index: string;
  models: number;
  score_histogram: HistogramBin[];
  saturation: Saturation | null;
  third_means: { top: number; middle: number; bottom: number };
  benchmark_discrimination: number;
  core_items: number;
  item_discrimination_histogram: HistogramBin[];
}

// How each benchmark spreads and separates the models (descriptive only; not used for scores).
export interface BenchmarkHealth {
  _meta: { params_id: string; subset_id: string; scope: string };
  definitions: Record<string, string>;
  tasks: Record<string, BenchmarkHealthEntry>;
}

export function getBenchmarkHealth(): BenchmarkHealth {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'benchmark_health.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8'));
}

function getLabels() {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'labels.json');
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  return JSON.parse(fileContents);
}

export function getTaxonomy(): Taxonomy {
  return getLabels()._schema;
}

export interface GroundingRef {
  framework: 'EU AI Act' | 'GPAI Code of Practice';
  citation: string;
  short: string;
  title: string;
  url: string;
  excerpt: string;
}

// direct: the cited text names what is measured. related: the measurement informs the cited risk.
// coverage "none": in the taxonomy's scope, but not measured by the current benchmarks.
export interface GroundingLink {
  id: string;
  strength: 'direct' | 'related';
  coverage?: 'none';
}

interface Grounding {
  _meta: {
    description: string;
    sources: Record<string, string>;
    paper: { title: string; url: string; year: number };
  };
  refs: Record<string, GroundingRef>;
  all: string[];
  indices: Record<string, string[]>;
  subcategories: Record<string, { refs: GroundingLink[]; lineage: string[] }>;
  // Cited risk sources and obligations that CORE does not currently evaluate.
  gaps: string[];
  // 2024 COMPL-AI technical requirements whose benchmarks are not in the CORE suite.
  requirements_not_in_core: { name: string; benchmarks: string[] }[];
}

export function getGrounding(): Grounding {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'grounding.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8'));
}

// Visitor-facing evaluation-area descriptions, keyed by taxonomy subcategory id.
export function getAreaDescriptions(): Record<string, string> {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'areas.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8')).areas;
}

export interface MethodLomoRow {
  model: string;
  index: 'capability' | 'reliability' | 'safety';
  observed: number;
  predicted: number;
  lower: number;
  upper: number;
  inside: boolean;
}

export interface MethodEciRow {
  model: string;
  type: string;
  ours: number;
  ours_lower: number;
  ours_upper: number;
  eci: number;
  eci_lower: number;
  eci_upper: number;
}

// One model in the methodology cost figure, built from costs.json.
export interface MethodCostRow {
  model: string;
  source: string;
  core: number;
  full: number;
  ratio: number;
}

export type MethodIndex = 'capability' | 'reliability' | 'safety';
interface MethodInterval { est: number; lo: number; hi: number }

// Every experiment number quoted on the methodology page; written with the figure rows by
// tools/method/agreement.py. Shares and errors are fractions (0.0147 = 1.47%); score errors are index points.
export interface MethodNumbers {
  panel: {
    models: number;
    closed_models: number;
    open_models: number;
    closed_names: string[];
    closed_tasks: Record<string, number>;
    responses: number;
    benchmarks: number;
    items: number;
    core_items: number;
    models_per_benchmark: { min: number; min_task: string; max: number; median: number };
  };
  lomo: {
    models: number;
    overall_mae: number;
    overall_mae_in_sample: number;
    task_mae_mean: number;
    task_mae_median: number;
    worst_task: string;
    worst_task_mae: number;
    max_drift_theta: number;
    index: Record<MethodIndex, {
      n: number;
      null: number;
      masked_mae: number;
      score_mae: number;
      score_max_ae: number;
      pairwise: number;
      spearman: number;
      coverage_90: number;
      coverage_90_in_sample: number;
    }>;
  };
  agreement: {
    eci: { n: number; pra: MethodInterval; pearson: MethodInterval };
    aa: { n: number; pra: MethodInterval; pearson: MethodInterval };
    ceiling: { n: number; pra: MethodInterval; pearson: MethodInterval; ours_vs_eci_pearson: MethodInterval };
    eci_pairs: { separated: number; separated_agree: number; overlap: number; overlap_agree: number };
    top3: { ours: string[]; eci: string[]; aa: string[] };
    top3_theta: Record<string, number>;
    eci_top15_missing: number;
    frontier: {
      k: number;
      eci_min: number;
      eci_max: number;
      pairs: number;
      separated: number;
      separated_agree: number;
      pra: MethodInterval;
      pearson: MethodInterval;
      aa_vs_eci: { n: number; pra: MethodInterval; pearson: MethodInterval };
      ours_vs_eci_same: MethodInterval;
      ours_vs_eci_same_pearson: MethodInterval;
    };
    distinct: Record<'reliability' | 'safety', { rho: number; n: number }>;
  };
}

interface AgreementData {
  _meta: Record<string, string>;
  lomo: MethodLomoRow[];
  eci: MethodEciRow[];
  numbers: MethodNumbers;
}

export function getAgreement(): AgreementData {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'agreement.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8'));
}

// The factor each index's published intervals are widened by, read from the predictions.
export function getIntervalScales(): Record<MethodIndex, number> {
  const models = Object.values(getPredictions().models ?? {}) as ModelPrediction[];
  const scale = (index: MethodIndex) =>
    models.map(m => m.indices?.[index]?.index_theta?.interval_scale).find(v => v != null) ?? 1;
  return { capability: scale('capability'), reliability: scale('reliability'), safety: scale('safety') };
}

// Leave-one-model-out rows with each interval widened around the prediction as published.
export function widenLomo(rows: MethodLomoRow[], scales: Record<MethodIndex, number>): MethodLomoRow[] {
  return rows.map(r => {
    const k = scales[r.index];
    const lower = r.predicted - k * (r.predicted - r.lower);
    const upper = r.predicted + k * (r.upper - r.predicted);
    return { ...r, lower, upper, inside: r.observed >= lower && r.observed <= upper };
  });
}

// Homepage results explorer (experiments.json). Model ids are prediction ids, as in JoinedModelData.id.
export interface ExplorerStep { id: string; date: string; score: number }

export interface ExplorerLag {
  open_steps: ExplorerStep[];
  closed_steps: ExplorerStep[];
  best_open: { id: string; score: number };
  best_closed: { id: string; score: number };
  gap: number;
  open_slope: number;
  lag_months: number;
  oldest_unmatched: { id: string; matched_by: string | null; months: number } | null;
  // Each proprietary frontier model and the first open model to match it (null if none has yet).
  observed: { id: string; matched_by: string | null; months: number }[];
}

export interface ExplorerLabStats {
  n: number;
  per_capability_point: number;
  per_month: number;
  rows: { id: string; lab: string; resid: number }[];
  labs: { lab: string; mean: number; median: number; n: number }[];
}

export interface ExplorerSafetyGroup {
  id: 'injection' | 'refusal';
  name: string;
  n: number;
  members: { id: string; name: string; pp_per_year: number }[];
  pp_per_year: number;
  pp_per_year_ci90: [number, number];
  trend: { date: string; y: number }[];
  points: { id: string; date: string; y: number }[];
}

export interface ExperimentsData {
  _meta: Record<string, string | number>;
  capability_safety: {
    labels: string[];
    safety: { rho: number; rho_release: number; n: number; top10_overlap: number };
    reliability: { rho: number; rho_release: number; n: number; top10_overlap: number };
  };
  safety_areas: {
    groups: ExplorerSafetyGroup[];
    boot: number;
  };
  open_lag: Record<'capability' | 'reliability' | 'safety', ExplorerLag> & { n_open: number; n_closed: number; as_of: string };
  params: {
    points: { id: string; total: number; active: number; moe: boolean; capability: number }[];
    n_points: number;
    n_points_moe: number;
    fit: {
      n: number;
      n_moe: number;
      per_10x_total: number;
      per_10x_active: number;
      per_10x_total_alone: number;
      per_10x_active_alone: number;
      median_release_moe: string;
      median_release_dense: string;
    };
  };
  labs: Record<'safety' | 'reliability', ExplorerLabStats>;
  frontier: Record<'capability' | 'reliability' | 'safety', { steps: ExplorerStep[]; start: ExplorerStep; end: ExplorerStep; delta: number; slope: number }> & {
    window: { from: string; to: string };
  };
}

export function getExperiments(): ExperimentsData {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'experiments.json');
  return JSON.parse(fs.readFileSync(dataPath, 'utf8'));
}

export function getBenchmarkLabels(): Record<string, BenchmarkLabels> {
  return Object.fromEntries(Object.entries(getLabels()).filter(([key]) => key !== '_schema')) as Record<string, BenchmarkLabels>;
}

// js-yaml follows YAML 1.2, which reads digit-grouped numbers like 128_000 as strings.
function normalizeGroupedNumbers(value: unknown): unknown {
  if (typeof value === 'string' && /^-?\d{1,3}(_\d{3})+(\.\d+)?$/.test(value)) {
    return Number(value.replace(/_/g, ''));
  }
  if (Array.isArray(value)) return value.map(normalizeGroupedNumbers);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, normalizeGroupedNumbers(item)])
    );
  }
  return value;
}

const BenchmarkMetaSchema = z.object({
  name: z.string().optional(),
  release_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).nullable().optional(),
  paper: z.string().url().nullable().optional(),
  repo: z.string().url().nullable().optional(),
  data_source: z
    .object({
      type: z.enum(['huggingface', 'github', 'url', 'inline']),
      id: z.string().optional(),
      url: z.string().url().optional(),
    })
    .nullable()
    .optional(),
});
export type BenchmarkMeta = z.infer<typeof BenchmarkMetaSchema>;

// Hand-maintained metadata from public/benchmarks.yaml, keyed by task id. Entries that fail
// validation are left out, so invalid values never reach the UI.
export function getBenchmarkMeta(): Record<string, BenchmarkMeta> {
  const dataPath = path.join(process.cwd(), 'public', 'benchmarks.yaml');
  if (!fs.existsSync(dataPath)) return {};
  // JSON_SCHEMA keeps unquoted dates like 2024-06-03 as strings instead of Date objects.
  const raw = (yaml.load(fs.readFileSync(dataPath, 'utf8'), { schema: yaml.JSON_SCHEMA }) ?? {}) as Record<string, unknown>;
  const meta: Record<string, BenchmarkMeta> = {};
  for (const [id, entry] of Object.entries(raw)) {
    const parsed = BenchmarkMetaSchema.safeParse(entry ?? {});
    if (parsed.success) {
      meta[id] = parsed.data;
      continue;
    }
    console.error(`\n❌ Zod Validation Error in benchmarks.yaml (${id}):`);
    parsed.error.issues.forEach(issue => console.error(`   - [${issue.path.join('.')}] ${issue.message}`));
  }
  return meta;
}

// Link to a benchmark's data: the Hugging Face dataset page, or the given URL.
export function dataSourceUrl(source: BenchmarkMeta['data_source']): string | null {
  if (!source) return null;
  if (source.url) return source.url;
  if (source.type === 'huggingface' && source.id) return `https://huggingface.co/datasets/${source.id}`;
  return null;
}

// Returns null for a file that fails validation, so invalid values never reach the UI.
function getModelMetadata(filename: string): ModelParams | null {
  const dataPath = path.join(process.cwd(), 'public', 'models', filename);
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  const parsed = ModelSchema.safeParse(normalizeGroupedNumbers(yaml.load(fileContents)));
  if (parsed.success) return parsed.data;

  console.error(`\n❌ Zod Validation Error in ${filename}:`);
  parsed.error.issues.forEach(issue => {
    console.error(`   - [${issue.path.join('.')}] ${issue.message}`);
  });
  return null;
}

function getAllModelMetadata(): Record<string, ModelParams> {
  const modelsDir = path.join(process.cwd(), 'public', 'models');
  if (!fs.existsSync(modelsDir)) return {};
  
  const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));
  const metadata: Record<string, ModelParams> = {};
  
  for (const file of files) {
    const id = file.replace(/\.ya?ml$/, '');
    const parsed = getModelMetadata(file);
    if (parsed) metadata[id] = parsed;
  }
  
  return metadata;
}

export function getJoinedModels(): JoinedModelData[] {
  const predictions = getPredictions();
  const costs = getCosts().models;
  const metadataMap = getAllModelMetadata();
  const joined: JoinedModelData[] = [];

  if (!predictions.models) return [];

  for (const [predId, predData] of Object.entries(predictions.models)) {
    let matchedYamlId: string | null = null;
    
    // Exact mapping lookup: Find the YAML file whose `evals` array contains this predId
    for (const [yamlId, meta] of Object.entries(metadataMap)) {
      if (meta.evals && Array.isArray(meta.evals) && meta.evals.includes(predId)) {
        matchedYamlId = yamlId;
        break;
      }
    }

    joined.push({
      id: predId,
      yamlId: matchedYamlId || predId.split('/').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '') || '',
      prediction: predData as ModelPrediction,
      runType: costs[predId]?.run_type ?? null,
      metadata: matchedYamlId ? metadataMap[matchedYamlId] : null,
    });
  }
  
  return joined;
}

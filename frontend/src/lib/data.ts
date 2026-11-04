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
  tasks: number;
  tasks_total: number;
}

// Scores are null when gap is set: the reason too few subset items cover the label.
// score_interval is a 90% bootstrap interval of predicted_score. It does not cover the
// error of imputing benchmarks the model did not run (imputed_items of population_items).
export interface LabelScore {
  gap: string | null;
  predicted_score: number | null;
  score_interval: [number, number] | null;
  population_items: number;
  population_tasks: number;
  subset_items: number;
  imputed_items: number;
  observations: number;
  ability: number | null;
  ability_standard_error: number | null;
  observed_subset_score: number | null;
  index_theta?: IndexTheta | null;
}

export type RunType = 'core' | 'full' | 'mixed';

// Tokens of the evaluated model (judge models are excluded; `judge` is kept for reference). `measured` is what the logs recorded;
// `core`/`full` are measured for the run that was done and extrapolated per item for the other.
interface ModelTokens {
  tasks: number;
  measured: number;
  input: number;
  output: number;
  reasoning: number;
  judge: number;
  core: number;
  full: number;
  reduction: number | null;
}

interface TaskUsage {
  // 'partial': more than the core items but well short of all items.
  run: 'core' | 'full' | 'partial';
  observed_items: number;
  subset_items: number;
  population_items: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  model_tokens: number;
  judge_tokens: number;
}

interface ModelUsage {
  run_type: RunType | null;
  observed_items: number;
  tasks?: Record<string, TaskUsage>;
  tokens: ModelTokens | null;
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
}

interface TaskEfficiency {
  subset_items: number;
  population_items: number;
  tokens_per_item_median: number;
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

export interface SuiteEfficiency {
  subset_items: number;
  population_items: number;
  core_tokens_median: number;
  full_tokens_median: number;
  token_reduction: number | null;
  run_types: Record<string, number>;
  tasks?: Record<string, TaskEfficiency>;
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
  usage?: ModelUsage;
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

// The newest eval across all models, so re-exports without new runs keep the same date.
export function getLastUpdated(): string | null {
  const dates = Object.values(getPredictions().models as Record<string, ModelPrediction>)
    .map(m => m.last_evaluated)
    .filter((d): d is string => !!d);
  return dates.length ? dates.reduce((a, b) => (a > b ? a : b)) : null;
}

export function getEfficiency(): SuiteEfficiency | null {
  return getPredictions().efficiency ?? null;
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

export interface MethodCostRow {
  model: string;
  source: string;
  core: number;
  full: number;
  ratio: number;
}

interface MethodData {
  _meta: Record<string, string>;
  lomo: MethodLomoRow[];
  eci: MethodEciRow[];
  cost: MethodCostRow[];
}

export function getMethod(): MethodData {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'method.json');
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
      metadata: matchedYamlId ? metadataMap[matchedYamlId] : null,
    });
  }
  
  return joined;
}

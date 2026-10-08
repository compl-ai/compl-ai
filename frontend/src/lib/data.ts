import fs from 'fs';
import path from 'path';
import * as yaml from 'js-yaml';

import { z } from 'zod';

export const ModelSchema = z.object({
  name: z.string(),
  release_date: z.string().nullable().optional(),
  open_weights: z.boolean().nullable().optional(),
  organization: z.string().nullable().optional(),
  license: z.string().nullable().optional(),
  
  availability: z.array(z.enum(["open-weights", "saas-chat", "api"])).nullable().optional(),
  description: z.string().nullable().optional(),
  
  specs: z.object({
    architecture: z.string().nullable().optional(),
    knowledge_cutoff: z.string().nullable().optional(),
    total_params: z.number().nullable().optional(),
    active_params: z.number().nullable().optional(),
    context: z.number().nullable().optional(),
    max_output_tokens: z.number().nullable().optional(),
    modalities: z.object({
      input: z.array(z.enum(["text", "image", "video"])).nullable().optional(),
      output: z.array(z.enum(["text", "image", "video"])).nullable().optional()
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

export type ModelParams = z.infer<typeof ModelSchema>;

// Index theta: the model's position on the index's scale (panel mean 0, SD 1),
// fitted on its scores for the index's benchmarks. Null when it ran too few of them.
export interface IndexTheta {
  theta: number;
  interval: [number, number];
  tasks: number;
  tasks_total: number;
}

// Scores are null when gap is set: the reason too few subset items cover the label.
export interface LabelScore {
  gap: string | null;
  predicted_score: number | null;
  population_items: number;
  population_tasks: number;
  subset_items: number;
  observations: number;
  ability: number | null;
  ability_standard_error: number | null;
  observed_subset_score: number | null;
  index_theta?: IndexTheta | null;
}

export interface ModelPrediction {
  predicted_score: number;
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
  tasks: {
    [task: string]: {
      predicted_score: number;
      observed_subset_score?: number;
      [key: string]: any;
    };
  };
}

export interface JoinedModelData {
  id: string; // The key from predictions (e.g. "openai-api/vllm/.../model")
  yamlId: string; // The inferred yaml key based on the name or ID
  prediction: ModelPrediction;
  metadata: ModelParams | null;
  // Debug artifact: per-task/label observed scores, plus a `_masked` detail object
  groundTruth?: Record<string, number | Record<string, unknown>> | null;
}

export function getPredictions() {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'predictions.core.v1.json');
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  return JSON.parse(fileContents);
}

export function getLabels() {
  const dataPath = path.join(process.cwd(), 'public', 'data', 'labels.json');
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  return JSON.parse(fileContents);
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
export function getModelMetadata(filename: string): ModelParams | null {
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

export function getAllModelMetadata(): Record<string, ModelParams> {
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
  const groundTruth = getGroundTruth();
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
      groundTruth: groundTruth ? groundTruth[predId] : null
    });
  }
  
  return joined;
}

export function getGroundTruth(): Record<string, Record<string, number>> | null {
  try {
    const p = path.join(process.cwd(), 'public', 'data', 'ground_truth.json');
    if (fs.existsSync(p)) {
      return JSON.parse(fs.readFileSync(p, 'utf8'));
    }
  } catch (e) {
    console.warn("Could not load ground truth data", e);
  }
  return null;
}

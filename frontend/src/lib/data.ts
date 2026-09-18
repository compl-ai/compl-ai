import fs from 'fs';
import path from 'path';
import * as yaml from 'js-yaml';

export interface ModelParams {
  name: string;
  release_date?: string;
  open_weights?: boolean;
  organization?: string;
  license?: string;
  specs?: {
    total_params?: number;
    active_params?: number;
    architecture?: string;
    context?: number;
  };
  [key: string]: any;
}

export interface ModelPrediction {
  predicted_score: number;
  predicted_score_error?: number;
  task_macro_score?: number;
  domains: {
    [domain: string]: number;
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

export function getModelMetadata(filename: string): ModelParams {
  const dataPath = path.join(process.cwd(), 'public', 'models', filename);
  const fileContents = fs.readFileSync(dataPath, 'utf8');
  return yaml.load(fileContents) as ModelParams;
}

export function getAllModelMetadata(): Record<string, ModelParams> {
  const modelsDir = path.join(process.cwd(), 'public', 'models');
  if (!fs.existsSync(modelsDir)) return {};
  
  const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.yaml') || f.endsWith('.yml'));
  const metadata: Record<string, ModelParams> = {};
  
  for (const file of files) {
    const id = file.replace(/\.ya?ml$/, '');
    metadata[id] = getModelMetadata(file);
  }
  
  return metadata;
}

export function getJoinedModels(): JoinedModelData[] {
  const predictions = getPredictions();
  const metadataMap = getAllModelMetadata();
  const joined: JoinedModelData[] = [];
  
  const normalizedMetadata = new Map<string, string>();
  for (const [yamlId, meta] of Object.entries(metadataMap)) {
    const normId = yamlId.toLowerCase().replace(/[^a-z0-9]/g, '');
    normalizedMetadata.set(normId, yamlId);
  }

  if (!predictions.models) return [];

  for (const [predId, predData] of Object.entries(predictions.models)) {
    const lastPart = predId.split('/').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '') || '';
    let matchedYamlId = normalizedMetadata.get(lastPart);
    
    if (!matchedYamlId) {
      for (const [normId, yamlId] of normalizedMetadata.entries()) {
        if (lastPart.includes(normId) || normId.includes(lastPart)) {
          matchedYamlId = yamlId;
          break;
        }
      }
    }

    joined.push({
      id: predId,
      yamlId: matchedYamlId || lastPart,
      prediction: predData as ModelPrediction,
      metadata: matchedYamlId ? metadataMap[matchedYamlId] : null
    });
  }
  
  return joined;
}

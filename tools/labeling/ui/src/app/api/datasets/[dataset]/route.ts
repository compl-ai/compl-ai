import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { NextResponse } from 'next/server';

const DATASETS_DIR = path.join(process.cwd(), '..', 'datasets');
const LABELS_DIR = path.join(process.cwd(), '..', 'labels');

export async function GET(
  request: Request,
  { params }: { params: Promise<{ dataset: string }> }
) {
  const resolvedParams = await params;
  const dataset = resolvedParams.dataset;
  
  try {
    const filePath = path.join(DATASETS_DIR, `${dataset}.jsonl`);
    const labelsPath = path.join(LABELS_DIR, `${dataset}.jsonl`);

    if (!fs.existsSync(filePath)) {
      return new NextResponse('Dataset not found', { status: 404 });
    }

    // Load labels into a map
    const labelsMap = new Map();
    if (fs.existsSync(labelsPath)) {
      const labelText = fs.readFileSync(labelsPath, 'utf-8');
      labelText.split('\n').forEach(line => {
        if (!line.trim()) return;
        try {
          const l = JSON.parse(line);
          labelsMap.set(l.sample_id, l);
        } catch (e) {}
      });
    }

    const rawText = fs.readFileSync(filePath, 'utf-8');
    const combinedLines = [];
    rawText.split('\n').forEach(line => {
        if (!line.trim()) return;
        try {
            const obj = JSON.parse(line);
            const labelObj = labelsMap.get(obj.sample_id);
            if (labelObj) {
                if (labelObj.llm_assigned) obj.llm_assigned = labelObj.llm_assigned;
                if (labelObj.deterministic_labels) obj.deterministic_labels = labelObj.deterministic_labels;
            }
            combinedLines.push(JSON.stringify(obj));
        } catch (e) {
            combinedLines.push(line);
        }
    });

    return new NextResponse(combinedLines.join('\n') + '\n', {
      headers: {
        'Content-Type': 'text/plain',
      },
    });
  } catch (error) {
    console.error(`Failed to read dataset ${dataset}:`, error);
    return new NextResponse('Failed to read dataset', { status: 500 });
  }
}

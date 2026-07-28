import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { NextResponse } from 'next/server';

const DATASETS_DIR = path.join(process.cwd(), '..', 'datasets');
const LABELS_DIR = path.join(process.cwd(), '..', 'labels');

export async function GET(
  request: Request,
  { params }: { params: Promise<{ dataset: string; sample_id: string }> }
) {
  const resolvedParams = await params;
  const { dataset, sample_id } = resolvedParams;
  
  try {
    const filePath = path.join(DATASETS_DIR, `${dataset}.jsonl`);
    const labelsPath = path.join(LABELS_DIR, `${dataset}.jsonl`);

    if (!fs.existsSync(filePath)) {
      return new NextResponse('Dataset not found', { status: 404 });
    }

    // Attempt to read the specific label
    let labelObj: any = null;
    if (fs.existsSync(labelsPath)) {
      const labelText = fs.readFileSync(labelsPath, 'utf-8');
      for (const line of labelText.split('\n')) {
        if (!line.trim()) continue;
        try {
          const l = JSON.parse(line);
          if (l.sample_id === sample_id) {
            labelObj = l;
            break;
          }
        } catch (e) {}
      }
    }

    // Stream through the dataset to find the exact row
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    for await (const line of rl) {
      if (!line.trim()) continue;
      
      // Fast check using regex first before JSON parse
      const match = line.match(/"sample_id"\s*:\s*"([^"]+)"/);
      if (match && match[1] === sample_id) {
        try {
          const obj = JSON.parse(line);
          if (labelObj) {
            if (labelObj.llm_assigned) obj.llm_assigned = labelObj.llm_assigned;
            if (labelObj.deterministic_labels) obj.deterministic_labels = labelObj.deterministic_labels;
          }
          return NextResponse.json(obj);
        } catch(e) {
          // Parse error, just return raw string wrapped in object
          return new NextResponse(line, { headers: { 'Content-Type': 'application/json' } });
        }
      }
    }

    return new NextResponse('Sample not found', { status: 404 });
  } catch (error) {
    console.error(`Failed to read sample ${sample_id} from ${dataset}:`, error);
    return new NextResponse('Failed to read sample', { status: 500 });
  }
}

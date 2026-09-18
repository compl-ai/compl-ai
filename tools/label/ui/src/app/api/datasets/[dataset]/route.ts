import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { Readable } from 'stream';
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

    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    const iterator = async function* () {
      for await (const line of rl) {
        if (!line.trim()) continue;
        let outputLine = line;
        
        // Strip out large base64 images to avoid crashing the browser with a 1GB payload
        outputLine = outputLine.replace(/"image"\s*:\s*"data:[^"]+;base64,[^"]+"/g, '"image":"[IMAGE_STRIPPED]"');
        
        try {
          const obj = JSON.parse(outputLine);
          const labelObj = labelsMap.get(obj.sample_id);
          if (labelObj) {
            if (labelObj.llm_assigned) obj.llm_assigned = labelObj.llm_assigned;
            if (labelObj.deterministic_labels) obj.deterministic_labels = labelObj.deterministic_labels;
            outputLine = JSON.stringify(obj);
          }
        } catch(e) {}

        yield outputLine + '\n';
      }
    };

    const stream = Readable.toWeb(Readable.from(iterator())) as ReadableStream;
    return new NextResponse(stream, { headers: { 'Content-Type': 'text/plain' } });
  } catch (error) {
    console.error(`Failed to read dataset ${dataset}:`, error);
    return new NextResponse('Failed to read dataset', { status: 500 });
  }
}

import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';

const LABELS_DIR = path.join(process.cwd(), '..', 'labels');

export async function GET(request: Request, { params }: { params: Promise<{ dataset: string }> }) {
  const resolvedParams = await params;
  const dataset = resolvedParams.dataset;
  const patchFilePath = path.join(LABELS_DIR, `${dataset}_patch.jsonl`);
  
  if (!existsSync(patchFilePath)) {
    return new NextResponse('', { headers: { 'Content-Type': 'text/plain' } });
  }

  const rawText = await fs.readFile(patchFilePath, 'utf-8');
  return new NextResponse(rawText, { headers: { 'Content-Type': 'text/plain' } });
}

export async function POST(request: Request, { params }: { params: Promise<{ dataset: string }> }) {
  const resolvedParams = await params;
  const dataset = resolvedParams.dataset;
  const patchFilePath = path.join(LABELS_DIR, `${dataset}_patch.jsonl`);
  
  try {
    const body = await request.json();
    const { sample_id, human_primary_label, human_secondary_labels, human_tags, human_rationale, human_approved } = body;

    const newPatch = {
      sample_id,
      human_primary_label,
      human_secondary_labels,
      human_tags,
      human_rationale,
      human_approved,
      timestamp: new Date().toISOString()
    };

    const patchMap = new Map();
    if (existsSync(patchFilePath)) {
      const rawText = await fs.readFile(patchFilePath, 'utf-8');
      rawText.split('\n').filter(l => l.trim()).forEach(l => {
        try {
          const p = JSON.parse(l);
          patchMap.set(p.sample_id, p);
        } catch (e) {}
      });
    }

    patchMap.set(sample_id, newPatch);

    const newContent = Array.from(patchMap.values()).map(p => JSON.stringify(p)).join('\n') + '\n';
    await fs.writeFile(patchFilePath, newContent, 'utf-8');
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to write patch:', error);
    return NextResponse.json({ error: 'Failed to write patch' }, { status: 500 });
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ dataset: string }> }) {
  const resolvedParams = await params;
  const dataset = resolvedParams.dataset;
  const patchFilePath = path.join(LABELS_DIR, `${dataset}_patch.jsonl`);
  
  try {
    const { searchParams } = new URL(request.url);
    const sample_id = searchParams.get('sample_id');
    
    if (!sample_id) {
      return NextResponse.json({ error: 'Missing sample_id' }, { status: 400 });
    }

    if (existsSync(patchFilePath)) {
      const patchMap = new Map();
      const rawText = await fs.readFile(patchFilePath, 'utf-8');
      
      rawText.split('\n').filter(l => l.trim()).forEach(l => {
        try {
          const p = JSON.parse(l);
          patchMap.set(String(p.sample_id), p);
        } catch (e) {}
      });

      if (patchMap.has(sample_id)) {
        patchMap.delete(sample_id);
        const newContent = patchMap.size > 0 
          ? Array.from(patchMap.values()).map(p => JSON.stringify(p)).join('\n') + '\n'
          : '';
          
        await fs.writeFile(patchFilePath, newContent, 'utf-8');
      }
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to delete patch:', error);
    return NextResponse.json({ error: 'Failed to delete patch' }, { status: 500 });
  }
}

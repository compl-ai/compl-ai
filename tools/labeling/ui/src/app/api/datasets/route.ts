import fs from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';

const DATASETS_DIR = path.join(process.cwd(), '..', 'datasets');

export async function GET() {
  try {
    const files = await fs.readdir(DATASETS_DIR);
    const datasets = [];
    
    for (const file of files) {
      if (file.endsWith('.jsonl')) {
        const filePath = path.join(DATASETS_DIR, file);
        const text = await fs.readFile(filePath, 'utf-8');
        const count = text.split('\n').filter(line => line.trim().length > 0).length;
        datasets.push({
          id: file.replace('.jsonl', ''),
          count
        });
      }
    }
      
    return NextResponse.json({ datasets });
  } catch (error) {
    console.error('Failed to read datasets directory:', error);
    return NextResponse.json({ error: 'Failed to read datasets' }, { status: 500 });
  }
}

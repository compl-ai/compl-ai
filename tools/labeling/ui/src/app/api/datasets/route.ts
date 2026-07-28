import fs from 'fs/promises';
import { createReadStream } from 'fs';
import path from 'path';
import readline from 'readline';
import { NextResponse } from 'next/server';

const DATASETS_DIR = path.join(process.cwd(), '..', 'datasets');

export async function GET() {
  try {
    const files = await fs.readdir(DATASETS_DIR);
    const datasets = [];
    
    for (const file of files) {
      if (file.endsWith('.jsonl')) {
        const filePath = path.join(DATASETS_DIR, file);
        
        let count = 0;
        const fileStream = createReadStream(filePath);
        const rl = readline.createInterface({
          input: fileStream,
          crlfDelay: Infinity
        });
        
        for await (const line of rl) {
          if (line.trim().length > 0) count++;
        }
        
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

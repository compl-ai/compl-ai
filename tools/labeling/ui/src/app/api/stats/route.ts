import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { NextResponse } from 'next/server';

const DATASETS_DIR = path.join(process.cwd(), '..', 'datasets');
const LABELS_DIR = path.join(process.cwd(), '..', 'labels');

export async function GET() {
  try {
    const files = fs.readdirSync(DATASETS_DIR).filter(f => f.endsWith('.jsonl'));
    
    const stats = {
      primaryLabels: {} as Record<string, number>,
      secondaryLabels: {} as Record<string, number>,
      confidence: { high: 0, medium: 0, low: 0 } as Record<string, number>,
      needsReview: 0,
      tags: {} as Record<string, number>,
      totalSamples: 0,
    };

    for (const file of files) {
      const labelsFilePath = path.join(LABELS_DIR, file);
      const patchFilePath = path.join(LABELS_DIR, file.replace('.jsonl', '_patch.jsonl'));
      
      const labelsMap = new Map();
      if (fs.existsSync(labelsFilePath)) {
        const labelsContent = fs.readFileSync(labelsFilePath, 'utf-8');
        labelsContent.split('\n').filter(l => l.trim()).forEach(l => {
          try {
            const p = JSON.parse(l);
            labelsMap.set(p.sample_id, p);
          } catch(e) {}
        });
      }

      const patchMap = new Map();
      if (fs.existsSync(patchFilePath)) {
        const patchContent = fs.readFileSync(patchFilePath, 'utf-8');
        patchContent.split('\n').filter(l => l.trim()).forEach(l => {
          try {
            const p = JSON.parse(l);
            patchMap.set(p.sample_id, p);
          } catch(e) {}
        });
      }

      const filePath = path.join(DATASETS_DIR, file);
      const fileStream = fs.createReadStream(filePath);
      const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
      });

      for await (const line of rl) {
        if (!line.trim()) continue;
        try {
          const row = JSON.parse(line);
          stats.totalSamples++;
          
          let llm = {};
          const labelObj = labelsMap.get(row.sample_id);
          if (labelObj && labelObj.llm_assigned) {
            llm = labelObj.llm_assigned;
          }
          
          const patch = patchMap.get(row.sample_id);
          
          let isReviewNeeded = !llm || !llm.primary_label || llm.label_confidence === 'low' || llm.primary_label === 'failed';
          
          if (patch) {
            if (patch.human_primary_label) llm.primary_label = patch.human_primary_label;
            if (patch.human_secondary_labels) llm.secondary_labels = patch.human_secondary_labels;
            if (patch.human_tags) llm.tags = patch.human_tags;
            isReviewNeeded = false; // Human reviewed it
          }

          if (isReviewNeeded && labelObj) { 
              // only count as needing review if it was actually labeled/processed, otherwise it's just unlabeled
              stats.needsReview++;
          }

          if (llm && llm.primary_label) {
            const conf = (llm.label_confidence || '').toLowerCase();
            if (conf === 'high' || conf === 'medium' || conf === 'low') {
              stats.confidence[conf]++;
            } else if (conf === 'mid') {
              stats.confidence['medium']++;
            }

            if (llm.primary_label) {
              stats.primaryLabels[llm.primary_label] = (stats.primaryLabels[llm.primary_label] || 0) + 1;
            }
            if (Array.isArray(llm.secondary_labels)) {
              for (const sl of llm.secondary_labels) {
                stats.secondaryLabels[sl] = (stats.secondaryLabels[sl] || 0) + 1;
              }
            }

            if (Array.isArray(llm.tags)) {
              for (const tag of llm.tags) {
                if (tag.startsWith('MODALITY:') || tag.startsWith('AGENT:')) {
                  stats.tags[tag] = (stats.tags[tag] || 0) + 1;
                }
              }
            }
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
    }

    return NextResponse.json(stats);
  } catch (error) {
    console.error('Failed to generate stats:', error);
    return NextResponse.json({ error: 'Failed to generate stats' }, { status: 500 });
  }
}

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { GoogleGenAI } from '@google/genai';
import * as dotenv from 'dotenv';
import { RateLimiter } from './rate-limiter';
import { llmAssignedGeminiSchema } from './types';
import { parse } from 'csv-parse/sync';

// Load .env
dotenv.config({ path: path.join(__dirname, '../.env') });
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
    console.error("❌ GEMINI_API_KEY not set");
    process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });
const limiter = new RateLimiter(5); // 5 concurrent requests

// Parse Args
const args = process.argv.slice(2);
let datasetName = '';
let limit = Infinity;
let mock = false;
let modelName = 'gemini-3.1-pro-preview';

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--dataset' && args[i+1]) {
        datasetName = args[i+1];
        i++;
    } else if (args[i] === '--limit' && args[i+1]) {
        limit = parseInt(args[i+1], 10);
        i++;
    } else if (args[i] === '--model' && args[i+1]) {
        modelName = args[i+1];
        i++;
    } else if (args[i] === '--mock') {
        mock = true;
    }
}

if (!datasetName) {
    console.error("❌ Usage: npx tsx labeler.ts --dataset <name> [--limit <num>] [--model <name>] [--mock]");
    process.exit(1);
}

// Paths
const datasetsDir = path.join(__dirname, '../datasets');
const labeledDir = path.join(__dirname, '../labels');
const inputFile = path.join(datasetsDir, `${datasetName}.jsonl`);
const outputFile = path.join(labeledDir, `${datasetName}.jsonl`);

if (!fs.existsSync(labeledDir)) {
    fs.mkdirSync(labeledDir, { recursive: true });
}

// Parse Taxonomy
function generateTaxonomyMd(csvPath: string) {
    console.log("Parsing taxonomy.csv...");
    const csvData = fs.readFileSync(csvPath, 'utf8');
    const records = parse(csvData, { columns: true, skip_empty_lines: true });

    const coreIndices = records.filter((r: any) => r.label_type === 'core_index');
    const itemsByCore: Record<string, any[]> = {};
    const tags: any[] = [];

    for (const row of records) {
        if (row.label_type === 'core_index') continue;
        if (row.label_id.startsWith('modality:') || row.label_id.startsWith('agent:')) {
            tags.push(row);
        } else {
            const c_idx = row.core_index;
            if (!itemsByCore[c_idx]) itemsByCore[c_idx] = [];
            itemsByCore[c_idx].push(row);
        }
    }

    const mdLines = ["# Taxonomy Labels", ""];

    for (const core of coreIndices) {
        mdLines.push(`## ${core.label_id} (${core.label_name})`);
        mdLines.push(core.description);
        mdLines.push("");

        const children = itemsByCore[core.label_id] || [];
        for (const child of children) {
            mdLines.push(`- **${child.label_id}** (${child.label_name})`);
            if (child.apply_when) mdLines.push(`  - *Apply when:* ${child.apply_when}`);
            if (child.do_not_apply_when) mdLines.push(`  - *Do not apply when:* ${child.do_not_apply_when}`);
        }
        mdLines.push("");
    }

    mdLines.push("## Tags (Modality & Agentic)", "");
    for (const tag of tags) {
        mdLines.push(`- **${tag.label_id}** (${tag.label_name})`);
        if (tag.apply_when) mdLines.push(`  - *Apply when:* ${tag.apply_when}`);
        if (tag.do_not_apply_when) mdLines.push(`  - *Do not apply when:* ${tag.do_not_apply_when}`);
        mdLines.push("");
    }

    const mdStr = mdLines.join('\n').trim() + '\n';
    fs.writeFileSync(path.join(__dirname, 'instructions/taxonomy_optimized.md'), mdStr, 'utf8');
    console.log("✅ Successfully parsed taxonomy.csv and generated taxonomy_optimized.md");
    return mdStr;
}

// Prepare System Instruction
const instructionsRaw = fs.readFileSync(path.join(__dirname, 'instructions/labeling_instructions.md'), 'utf8');
const taxonomyMd = generateTaxonomyMd(path.join(__dirname, '../taxonomy.csv'));
const systemInstruction = instructionsRaw.replace('{{INJECT_TAXONOMY_OPTIMIZED_MD_HERE}}', taxonomyMd);

// Load existing sample_ids for Pause/Resume
const completedSampleIds = new Set<string>();
if (fs.existsSync(outputFile)) {
    const lines = fs.readFileSync(outputFile, 'utf8').split('\n');
    for (const line of lines) {
        if (!line.trim()) continue;
        try {
            const obj = JSON.parse(line);
            if (obj.sample_id) completedSampleIds.add(obj.sample_id);
        } catch (e) {}
    }
}
console.log(`✅ Loaded ${completedSampleIds.size} existing samples from ${outputFile}`);
console.log(`🤖 Using model: ${modelName}`);

import { formatGeminiParts } from '../ui/src/lib/multimodal';

async function processSample(lineObj: any): Promise<any> {
    const promptData = {
        input: lineObj.input,
        target: lineObj.target,
        metadata: lineObj.metadata,
        deterministic_labels: lineObj.deterministic_labels || []
    };
    
    // Auto-append benchmark label
    const benchmarkLabel = `benchmark:${datasetName}`;
    if (!promptData.deterministic_labels.includes(benchmarkLabel)) {
        promptData.deterministic_labels.push(benchmarkLabel);
    }
    lineObj.deterministic_labels = promptData.deterministic_labels; // mutate original to save it later
    
    // Dry run mock (skip LLM)
    if (modelName === 'mock' || mock) {
        return new Promise(resolve => setTimeout(() => resolve({
            primary_label: "safety",
            secondary_labels: ["safety:harmful-instruction-refusal"],
            tags: ["modality:static-mcq"],
            label_confidence: "high",
            needs_human_review: false,
            ambiguity_reason: null,
            label_rationale: "MOCK REASON: Just testing the pipeline."
        }), 500));
    }

    const inputParts = formatGeminiParts(promptData.input);
    const contextJson = JSON.stringify({
        target: promptData.target,
        metadata: promptData.metadata,
        deterministic_labels: promptData.deterministic_labels
    }, null, 2);

    const geminiContents = [
        { text: "Here is the JSON sample. The `input` field is provided separately below, including any images:\n```json\n" + contextJson + "\n```\n\n### Input Prompt:\n" },
        ...inputParts
    ];

    return limiter.run(async () => {
        const response = await Promise.race([
            ai.models.generateContent({
                model: modelName,
                config: {
                    systemInstruction: systemInstruction,
                    responseMimeType: "application/json",
                    responseSchema: llmAssignedGeminiSchema,
                    temperature: 0.2,
                    safetySettings: [
                        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" }
                    ]
                },
                contents: [{ role: 'user', parts: geminiContents }]
            }),
            new Promise<any>((_, reject) => setTimeout(() => reject(new Error("API timeout after 30s")), 30000))
        ]);
        
        let text;
        try {
            text = response.text;
        } catch (e) {
            // Getter throws if blocked
        }

        if (!text) {
            return {
                primary_label: "safety",
                secondary_labels: ["safety:harmful-instruction-refusal"],
                tags: [],
                label_confidence: "low",
                needs_human_review: true,
                ambiguity_reason: "API_SAFETY_BLOCK",
                label_rationale: "Gemini API blocked this prompt entirely at the safety filter level."
            };
        }
        
        return JSON.parse(text);
    });
}

async function main() {
    if (!fs.existsSync(inputFile)) {
        console.error(`❌ Input file not found: ${inputFile}`);
        process.exit(1);
    }

    const fileStream = fs.createReadStream(inputFile);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    let processedNew = 0;
    const promises: Promise<void>[] = [];

    for await (const line of rl) {
        if (!line.trim()) continue;
        
        let obj;
        try {
            obj = JSON.parse(line);
        } catch (e) {
            console.error("Skipping invalid JSON line");
            continue;
        }

        if (!obj.sample_id) {
            console.warn("⚠️ Sample missing sample_id. Skipping.");
            continue;
        }

        if (completedSampleIds.has(obj.sample_id)) {
            continue; // Skip already completed
        }

        if (processedNew >= limit) {
            break;
        }

        processedNew++;

        const p = processSample(obj)
            .then(llmResult => {
                obj.llm_assigned = llmResult;
                const labelObj = {
                    sample_id: obj.sample_id,
                    deterministic_labels: obj.deterministic_labels || [],
                    llm_assigned: obj.llm_assigned || {}
                };
                fs.appendFileSync(outputFile, JSON.stringify(labelObj) + '\n');
                completedSampleIds.add(obj.sample_id);
                console.log(`✅ Labeled sample_id: ${obj.sample_id} -> ${llmResult.primary_label}`);
            })
            .catch(error => {
                console.error(`❌ Failed sample_id: ${obj.sample_id} -`, error);
                obj.llm_assigned = {
                    primary_label: "failed",
                    secondary_labels: [],
                    tags: [],
                    label_confidence: "low",
                    needs_human_review: true,
                    ambiguity_reason: "SYSTEM_ERROR",
                    label_rationale: `Script Error: ${error.message}`
                };
                const labelObj = {
                    sample_id: obj.sample_id,
                    deterministic_labels: obj.deterministic_labels || [],
                    llm_assigned: obj.llm_assigned || {}
                };
                fs.appendFileSync(outputFile, JSON.stringify(labelObj) + '\n');
                completedSampleIds.add(obj.sample_id);
            });
            
        promises.push(p);
    }

    // Wait for all queued requests to finish
    await Promise.all(promises);

    console.log(`\n🎉 Run complete! Successfully processed ${processedNew} new samples.`);
    process.exit(0);
}

main();

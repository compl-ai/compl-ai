import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { GoogleGenAI } from '@google/genai';
import Anthropic from '@anthropic-ai/sdk';
import * as dotenv from 'dotenv';
import { RateLimiter } from './rate-limiter';
import { buildAnthropicSchema, buildGeminiSchema, TaxonomyLabelSets } from './types';
import { parse } from 'csv-parse/sync';
import { formatGeminiParts } from '../ui/src/lib/multimodal';

dotenv.config({ path: path.join(__dirname, '../.env') });

// Parse Args
const USAGE = "❌ Usage: npx tsx labeler.ts --dataset <name> [--provider gemini|anthropic] [--model <name>] [--limit <num>] [--relabel] [--mock]";
const args = process.argv.slice(2);
let datasetName = '';
let limit = Infinity;
let mock = false;
let relabel = false;
let modelName = '';
let provider = '';

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--dataset' && args[i+1]) {
        datasetName = args[++i];
    } else if (args[i] === '--limit' && args[i+1]) {
        limit = parseInt(args[++i], 10);
    } else if (args[i] === '--model' && args[i+1]) {
        modelName = args[++i];
    } else if (args[i] === '--provider' && args[i+1]) {
        provider = args[++i];
    } else if (args[i] === '--mock') {
        mock = true;
    } else if (args[i] === '--relabel') {
        relabel = true;
    }
}

if (!datasetName) {
    console.error(USAGE);
    process.exit(1);
}

const DEFAULT_MODELS: Record<string, string> = {
    gemini: 'gemini-3.1-pro-preview',
    anthropic: 'claude-opus-5-5',
};
if (!provider) provider = modelName.startsWith('claude') ? 'anthropic' : 'gemini';
if (!(provider in DEFAULT_MODELS)) {
    console.error(`❌ Unknown provider '${provider}'.\n${USAGE}`);
    process.exit(1);
}
if (!modelName) modelName = DEFAULT_MODELS[provider];
if (modelName === 'mock') mock = true;

let gemini: GoogleGenAI | undefined;
let anthropic: Anthropic | undefined;
if (!mock) {
    const keyName = provider === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'GEMINI_API_KEY';
    const apiKey = process.env[keyName];
    if (!apiKey) {
        console.error(`❌ ${keyName} not set (add it to tools/label/.env)`);
        process.exit(1);
    }
    if (provider === 'anthropic') anthropic = new Anthropic({ apiKey, maxRetries: 0 });
    else gemini = new GoogleGenAI({ apiKey });
    // Claude's API safety classifier refuses some red-teaming prompts (e.g. malware requests); fall back to Gemini for those.
    if (provider === 'anthropic' && process.env.GEMINI_API_KEY) {
        gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    }
}
const FALLBACK_MODEL = DEFAULT_MODELS.gemini;
const BLOCKED_REASONS = new Set(['API_REFUSAL', 'API_SAFETY_BLOCK']);

const limiter = new RateLimiter(5); // 5 concurrent requests
const API_TIMEOUT_MS = provider === 'anthropic' ? 120000 : 30000;

// Paths
const datasetsDir = path.join(__dirname, '../datasets');
const labeledDir = path.join(__dirname, '../labels');
const inputFile = path.join(datasetsDir, `${datasetName}.jsonl`);
const outputFile = path.join(labeledDir, `${datasetName}.jsonl`);
const patchFile = path.join(labeledDir, `${datasetName}_patch.jsonl`);

if (!fs.existsSync(labeledDir)) {
    fs.mkdirSync(labeledDir, { recursive: true });
}

// --relabel: move existing LLM labels and human patches aside so the run starts fresh and
// the review UI shows the new labels. The archive is a subfolder so loaders (which glob
// labels/*.jsonl non-recursively) ignore it.
if (relabel) {
    const archiveDir = path.join(labeledDir, 'archive');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    fs.mkdirSync(archiveDir, { recursive: true });
    for (const [file, suffix] of [[outputFile, ''], [patchFile, '_patch']] as const) {
        if (fs.existsSync(file)) {
            const dest = path.join(archiveDir, `${datasetName}.${stamp}${suffix}.jsonl`);
            fs.renameSync(file, dest);
            console.log(`📦 Archived ${path.basename(file)} -> ${path.relative(labeledDir, dest)}`);
        }
    }
}

// Parse Taxonomy
function generateTaxonomyMd(records: any[]) {
    const coreIndices = records.filter((r: any) => r.label_type === 'core_index');
    const itemsByCore: Record<string, any[]> = {};
    const tags: any[] = [];

    for (const row of records) {
        if (row.label_type === 'core_index' || row.label_type === 'subcategory') continue;
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

console.log("Parsing taxonomy.csv...");
const taxonomyRecords: any[] = parse(fs.readFileSync(path.join(__dirname, '../../../src/complai/data/taxonomy.csv'), 'utf8'), { columns: true, skip_empty_lines: true });
const labelSets: TaxonomyLabelSets = {
    secondary: taxonomyRecords
        .filter(r => r.label_type !== 'core_index' && r.label_type !== 'subcategory' && !r.label_id.startsWith('modality:') && !r.label_id.startsWith('agent:'))
        .map(r => r.label_id),
    tags: taxonomyRecords
        .filter(r => r.label_id.startsWith('modality:') || r.label_id.startsWith('agent:'))
        .map(r => r.label_id),
};
const taxonomyMd = generateTaxonomyMd(taxonomyRecords);

// Prepare System Instruction
const instructionsRaw = fs.readFileSync(path.join(__dirname, 'instructions/labeling_instructions.md'), 'utf8');
const systemInstruction = instructionsRaw.replace('{{INJECT_TAXONOMY_OPTIMIZED_MD_HERE}}', taxonomyMd);

const geminiSchema = buildGeminiSchema(labelSets);
const anthropicSchema = buildAnthropicSchema(labelSets);

// Load existing sample_ids for Pause/Resume. Refused/blocked rows are dropped and retried.
const completedSampleIds = new Set<string>();
if (fs.existsSync(outputFile)) {
    const lines = fs.readFileSync(outputFile, 'utf8').split('\n');
    const kept: string[] = [];
    let retrying = 0;
    for (const line of lines) {
        if (!line.trim()) continue;
        try {
            const obj = JSON.parse(line);
            if (!mock && BLOCKED_REASONS.has(obj.llm_assigned?.ambiguity_reason)) {
                retrying++;
                continue;
            }
            if (obj.sample_id && obj.llm_assigned?.primary_label !== 'failed') {
                completedSampleIds.add(obj.sample_id);
            }
        } catch (e) {}
        kept.push(line);
    }
    if (retrying > 0) {
        fs.writeFileSync(outputFile, kept.map(l => l + '\n').join(''));
        console.log(`🔁 Retrying ${retrying} previously refused/blocked samples`);
    }
}
console.log(`✅ Loaded ${completedSampleIds.size} existing samples from ${outputFile}`);
console.log(`🤖 Using ${mock ? 'MOCK' : provider} model: ${modelName}`);

function blockedResult(reason: string, rationale: string) {
    return {
        primary_label: "safety",
        secondary_labels: ["safety:harmful-instruction-refusal"],
        tags: [],
        label_confidence: "low",
        needs_human_review: true,
        ambiguity_reason: reason,
        label_rationale: rationale,
    };
}

function withTimeout<T>(p: Promise<T>): Promise<T> {
    return Promise.race([
        p,
        new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`API timeout after ${API_TIMEOUT_MS / 1000}s`)), API_TIMEOUT_MS)),
    ]);
}

function toAnthropicContent(geminiParts: any[]): Anthropic.ContentBlockParam[] {
    return geminiParts.map(part => part.inlineData
        ? { type: 'image', source: { type: 'base64', media_type: part.inlineData.mimeType, data: part.inlineData.data } }
        : { type: 'text', text: part.text }
    ) as Anthropic.ContentBlockParam[];
}

async function callGemini(contentParts: any[], model: string = modelName): Promise<any> {
    const response = await withTimeout(gemini!.models.generateContent({
        model,
        config: {
            systemInstruction: systemInstruction,
            responseMimeType: "application/json",
            responseSchema: geminiSchema,
            temperature: 0.2,
            safetySettings: [
                { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
                { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
                { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
                { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" }
            ] as any
        },
        contents: [{ role: 'user', parts: contentParts }]
    }));

    let text;
    try {
        text = response.text;
    } catch (e) {
        // Getter throws if blocked
    }
    if (!text) {
        return blockedResult("API_SAFETY_BLOCK", "Gemini API blocked this prompt entirely at the safety filter level.");
    }
    return JSON.parse(text);
}

async function callAnthropic(contentParts: any[]): Promise<any> {
    const response = await withTimeout(anthropic!.messages.create({
        model: modelName,
        max_tokens: 2048,
        // Cache the (large, identical) system prompt across samples.
        system: [{ type: 'text', text: systemInstruction, cache_control: { type: 'ephemeral' } }],
        output_config: { format: { type: 'json_schema', schema: anthropicSchema } },
        messages: [{ role: 'user', content: toAnthropicContent(contentParts) }],
    }));

    if (response.stop_reason === 'refusal') {
        return blockedResult("API_REFUSAL", `${modelName} refused to label this sample.`);
    }
    if (response.stop_reason === 'max_tokens') {
        throw new Error("Anthropic response truncated (max_tokens); JSON is incomplete");
    }
    const text = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === 'text')
        .map(b => b.text)
        .join('');
    return JSON.parse(text);
}

// Schema enforcement guarantees valid label ids; this checks the cross-field rule the schema can't express.
function checkConsistency(result: any): any {
    const first = result.secondary_labels?.[0];
    if (first && result.primary_label && !first.startsWith(`${result.primary_label}:`)) {
        const note = `First secondary '${first}' is not under primary '${result.primary_label}'.`;
        console.warn(`⚠️  ${note}`);
        result.needs_human_review = true;
        result.ambiguity_reason = result.ambiguity_reason ? `${result.ambiguity_reason} ${note}` : note;
    }
    return result;
}

const refusedIds: string[] = [];
const fallbackIds: string[] = [];

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
    if (mock) {
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

    const contextJson = JSON.stringify({
        target: promptData.target,
        metadata: promptData.metadata,
        deterministic_labels: promptData.deterministic_labels
    }, null, 2);

    const contentParts = [
        { text: "Here is the JSON sample. The `input` field is provided separately below, including any images:\n```json\n" + contextJson + "\n```\n\n### Input Prompt:\n" },
        ...formatGeminiParts(promptData.input)
    ];

    return limiter.run(async () => {
        if (provider !== 'anthropic') return checkConsistency({ ...(await callGemini(contentParts)), model: modelName });
        const result = await callAnthropic(contentParts);
        if (result.ambiguity_reason === 'API_REFUSAL' && gemini) {
            console.warn(`🚫 ${modelName} refused sample_id: ${lineObj.sample_id} -> retrying with ${FALLBACK_MODEL}`);
            const fallback = await callGemini(contentParts, FALLBACK_MODEL);
            if (!BLOCKED_REASONS.has(fallback.ambiguity_reason)) fallbackIds.push(lineObj.sample_id);
            return checkConsistency({ ...fallback, model: FALLBACK_MODEL });
        }
        return checkConsistency({ ...result, model: modelName });
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
                obj.llm_assigned = { ...llmResult, model: mock ? 'mock' : (llmResult.model ?? modelName) };
                const labelObj = {
                    sample_id: obj.sample_id,
                    deterministic_labels: obj.deterministic_labels || [],
                    llm_assigned: obj.llm_assigned
                };
                fs.appendFileSync(outputFile, JSON.stringify(labelObj) + '\n');
                completedSampleIds.add(obj.sample_id);
                if (BLOCKED_REASONS.has(llmResult.ambiguity_reason)) {
                    refusedIds.push(obj.sample_id);
                    console.warn(`🚫 REFUSED sample_id: ${obj.sample_id} (${llmResult.ambiguity_reason}) - saved as needs-review; re-run to retry`);
                    return;
                }
                console.log(`✅ Labeled sample_id: ${obj.sample_id} -> primary: ${llmResult.primary_label} | secondary: [${(llmResult.secondary_labels || []).join(', ')}]${fallbackIds.includes(obj.sample_id) ? `  (via ${FALLBACK_MODEL} fallback)` : ''}`);
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
                    llm_assigned: obj.llm_assigned
                };
                fs.appendFileSync(outputFile, JSON.stringify(labelObj) + '\n');
                completedSampleIds.add(obj.sample_id);
            });

        promises.push(p);
    }

    // Wait for all queued requests to finish
    await Promise.all(promises);

    if (fallbackIds.length > 0) {
        console.warn(`\n🔀 ${fallbackIds.length} sample(s) refused by ${modelName} and labeled by ${FALLBACK_MODEL} instead: ${fallbackIds.join(', ')}`);
    }
    if (refusedIds.length > 0) {
        console.warn(`🚫 ${refusedIds.length} sample(s) still refused/blocked (saved as low-confidence, needs review): ${refusedIds.join(', ')}`);
        console.warn(`   Re-running the same command retries them.`);
    }

    console.log(`\n🎉 Run complete! Successfully processed ${processedNew} new samples.`);
    process.exit(0);
}

main();

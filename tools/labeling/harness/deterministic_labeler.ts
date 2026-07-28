import fs from 'fs';
import path from 'path';
import readline from 'readline';

// Paths
const datasetsDir = path.join(__dirname, '../datasets');
const labeledDir = path.join(__dirname, '../labels');
const rulesFile = path.join(__dirname, 'deterministic_rules.json');

// Parse Args
const args = process.argv.slice(2);
let datasetName = '';

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--dataset' && args[i+1]) {
        datasetName = args[i+1];
        i++;
    }
}

if (!datasetName) {
    console.error("❌ Usage: npx tsx deterministic_labeler.ts --dataset <name>");
    process.exit(1);
}

// Ensure output directory exists
if (!fs.existsSync(labeledDir)) {
    fs.mkdirSync(labeledDir, { recursive: true });
}

// Load Rules
if (!fs.existsSync(rulesFile)) {
    console.error(`❌ Rules file not found at ${rulesFile}`);
    process.exit(1);
}
const rulesObj = JSON.parse(fs.readFileSync(rulesFile, 'utf8'));
const datasetRules = rulesObj.datasets[datasetName];

if (!datasetRules) {
    console.error(`❌ No rules found for dataset '${datasetName}' in deterministic_rules.json`);
    process.exit(1);
}

const inputFile = path.join(datasetsDir, `${datasetName}.jsonl`);
const outputFile = path.join(labeledDir, `${datasetName}.jsonl`);

if (!fs.existsSync(inputFile)) {
    console.error(`❌ Input file not found: ${inputFile}`);
    process.exit(1);
}

console.log(`🚀 Deterministically labeling ${datasetName}...`);

const fileStream = fs.createReadStream(inputFile);
const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
const outStream = fs.createWriteStream(outputFile);

let count = 0;

rl.on('line', (line) => {
    if (!line.trim()) return;
    try {
        const obj = JSON.parse(line);
        
        let primary = datasetRules.primary_label || "";
        let secondary = datasetRules.secondary_labels ? [...datasetRules.secondary_labels] : [];
        let tags = datasetRules.tags ? [...datasetRules.tags] : [];

        // Advanced: Metadata-based override logic
        if (datasetRules.metadata_rules) {
            const field = datasetRules.metadata_rules.field;
            const mapping = datasetRules.metadata_rules.mapping;
            const val = obj.metadata?.[field];
            
            const match = mapping[val] || mapping["default"];
            if (match) {
                if (match.primary_label) primary = match.primary_label;
                if (match.secondary_labels) secondary = match.secondary_labels;
                if (match.tags) tags = match.tags;
            }
        }

        // Auto-append benchmark label
        const benchmarkLabel = `benchmark:${datasetName}`;
        obj.deterministic_labels = obj.deterministic_labels || [];
        if (!obj.deterministic_labels.includes(benchmarkLabel)) {
            obj.deterministic_labels.push(benchmarkLabel);
        }
        
        // Construct the llm_assigned object to mimic the AI outputs
        obj.llm_assigned = {
            primary_label: primary,
            secondary_labels: secondary,
            tags: tags,
            label_confidence: "high",
            needs_human_review: false,
            reasoning: "Deterministically labeled based on dataset configuration."
        };
        
        const labelObj = {
            sample_id: obj.sample_id,
            deterministic_labels: obj.deterministic_labels || [],
            llm_assigned: obj.llm_assigned || {}
        };
        
        outStream.write(JSON.stringify(labelObj) + '\n');
        count++;
    } catch (e) {
        console.error("Error parsing line:", e);
    }
});

rl.on('close', () => {
    outStream.end();
    console.log(`✅ Finished writing ${count} labeled samples to ${outputFile}`);
});

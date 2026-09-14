import { GoogleGenAI } from '@google/genai';
import { formatGeminiParts } from '../ui/src/lib/multimodal';
import fs from 'fs';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const line = fs.readFileSync('datasets/mmmu_pro.jsonl', 'utf8').split('\n')[0];
const obj = JSON.parse(line);
const parts = formatGeminiParts(obj.input);

console.log("Parts:", parts.map(p => p.inlineData ? `image (${p.inlineData.mimeType}, ${p.inlineData.data.substring(0, 20)}...)` : `text (${p.text?.substring(0, 50)})`));

async function run() {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-3.5-flash',
            contents: [{ role: 'user', parts: parts }]
        });
        console.log(response.text);
    } catch (e: any) {
        console.error("Error:", e);
        if (e.cause) console.error("Cause:", e.cause);
    }
}
run();

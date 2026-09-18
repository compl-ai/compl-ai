import { Schema, Type } from '@google/genai';

// Gemini API expects an OpenAPI 3.0 Schema object for structured outputs
export const llmAssignedGeminiSchema: Schema = {
    type: Type.OBJECT,
    properties: {
        primary_label: {
            type: Type.STRING,
            enum: ["safety", "security-privacy", "fairness-bias", "reliability", "capability"],
            description: "The top-level core benchmark index (e.g. safety, capability)",
        },
        secondary_labels: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "Additional relevant sub-labels from the taxonomy that describe the sample.",
        },
        tags: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "Modality or agentic properties of the sample.",
        },
        label_confidence: {
            type: Type.STRING,
            enum: ["high", "medium", "low"],
            description: "Your confidence in the primary label.",
        },
        needs_human_review: {
            type: Type.BOOLEAN,
            description: "Set to true if confidence is low, highly ambiguous, or multiple primary labels seem equally valid.",
        },
        ambiguity_reason: {
            type: Type.STRING,
            description: "Explanation if human review is needed, otherwise null.",
            nullable: true,
        },
        label_rationale: {
            type: Type.STRING,
            description: "Extremely brief explanation (under 25 words) of why you chose the primary label.",
        }
    },
    required: ["primary_label", "secondary_labels", "label_confidence", "needs_human_review", "label_rationale"],
};

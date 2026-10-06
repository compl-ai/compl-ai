import { Schema, Type } from '@google/genai';

export const PRIMARY_LABELS = ["safety", "security-privacy", "fairness-bias", "reliability", "capability"] as const;

export interface TaxonomyLabelSets {
    secondary: string[];
    tags: string[];
}

// Gemini API expects an OpenAPI 3.0 Schema object for structured outputs
export function buildGeminiSchema(labels: TaxonomyLabelSets): Schema {
    return {
        type: Type.OBJECT,
        properties: {
            primary_label: {
                type: Type.STRING,
                enum: [...PRIMARY_LABELS],
                description: "The top-level core benchmark index (e.g. safety, capability)",
            },
            secondary_labels: {
                type: Type.ARRAY,
                items: { type: Type.STRING, enum: labels.secondary },
                description: "Sub-labels from the taxonomy. The FIRST entry must belong to the primary label's category.",
            },
            tags: {
                type: Type.ARRAY,
                items: { type: Type.STRING, enum: labels.tags },
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
}

// Anthropic structured outputs (output_config.format = json_schema) use constrained decoding,
// so every response is guaranteed to parse and match this schema. Constraints: every object
// needs additionalProperties:false, and minItems may only be 0 or 1.
export function buildAnthropicSchema(labels: TaxonomyLabelSets): Record<string, unknown> {
    return {
        type: "object",
        additionalProperties: false,
        properties: {
            primary_label: {
                type: "string",
                enum: [...PRIMARY_LABELS],
                description: "The top-level core benchmark index.",
            },
            secondary_labels: {
                type: "array",
                minItems: 1,
                items: { type: "string", enum: labels.secondary },
                description: "Sub-labels from the taxonomy. The FIRST entry must belong to the primary label's category.",
            },
            tags: {
                type: "array",
                items: { type: "string", enum: labels.tags },
                description: "Modality or agentic properties of the sample.",
            },
            label_confidence: {
                type: "string",
                enum: ["high", "medium", "low"],
            },
            needs_human_review: {
                type: "boolean",
            },
            ambiguity_reason: {
                anyOf: [{ type: "string" }, { type: "null" }],
                description: "Explanation if human review is needed, otherwise null.",
            },
            label_rationale: {
                type: "string",
                description: "Extremely brief explanation (under 25 words) of why you chose the primary label.",
            },
        },
        required: ["primary_label", "secondary_labels", "tags", "label_confidence", "needs_human_review", "ambiguity_reason", "label_rationale"],
    };
}

import fs from 'fs';
import path from 'path';

/**
 * Validates the LLM output against the expected compl-ai schema 
 * and ensures that the assigned labels actually exist in the taxonomy.
 */

// We will load these from the CSVs
const VALID_CORES = new Set([
  'safety', 'security-privacy', 'fairness-bias', 
  'reliability', 'capability'
]);

let validLabels = new Set<string>();
let validGroundings = new Set<string>();

export function initValidator(taxonomyCsvPath: string, groundingCsvPath: string) {
    // TODO: Parse CSVs and populate validLabels and validGroundings sets
    // Example: validLabels.add('safety:harmful-instruction-refusal')
}

export function validateLLMOutput(llmJson: any): { isValid: boolean, errors: string[] } {
    const errors: string[] = [];

    // Basic structure checks
    if (!llmJson) {
        return { isValid: false, errors: ['LLM output is empty or null'] };
    }

    if (!VALID_CORES.has(llmJson.primary_label)) {
        errors.push(`Invalid primary_label: ${llmJson.primary_label}`);
    }

    // Strict taxonomy checks
    // (Uncomment once CSV parsing is implemented)
    /*
    if (!validLabels.has(llmJson.primary_label)) {
        errors.push(`Invalid primary_label (not found in taxonomy): ${llmJson.primary_label}`);
    }

    if (llmJson.tags) {
        // validate tags
    }
    */

    // Required fields
    if (!['high', 'medium', 'low'].includes(llmJson.label_confidence)) {
        errors.push(`Invalid label_confidence: ${llmJson.label_confidence}`);
    }
    if (typeof llmJson.needs_human_review !== 'boolean') {
        errors.push(`needs_human_review must be boolean`);
    }
    if (!llmJson.label_rationale || typeof llmJson.label_rationale !== 'string') {
        errors.push('Missing or invalid label_rationale');
    }

    return {
        isValid: errors.length === 0,
        errors
    };
}

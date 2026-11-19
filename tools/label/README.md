# Compl-AI Labeling Pipeline

This directory contains the infrastructure for generating, reviewing, and managing metadata labels (e.g., safety, bias, capabilities) for the `compl-ai` benchmark datasets.

## Architecture & Data Storage (Stand-off Annotations)

To avoid polluting the original data and to prevent large git history issues, we use a **stand-off annotation** model:

1. **`datasets/`**: Contains the original, unmodified `.jsonl` benchmark datasets. These are **NOT** checked into git. You must generate them locally using `cd datasets && ./regenerate_datasets.sh`.
2. **`labels/`**: Contains only the labels, metadata, and human review patches. These files (`.jsonl`) are tracked in Git. The UI and analysis scripts join them with the raw datasets at runtime using the `sample_id`.

## Pipeline Components & Folders

- **`datasets/`**: The un-labeled, raw evaluation datasets. Must be generated locally (see `regenerate_datasets.sh`).
- **`labels/`**: The resulting `.jsonl` files containing taxonomy metadata and human patches. These are committed to Git.
- **`ui/`**: A Next.js dashboard used by human annotators to review the labels. (See `ui/README.md` for startup instructions).
- **`harness/`**: The automated LLM-based labeling scripts. 

### Running the Labeler (`harness/`)

To automatically label a dataset using an LLM, navigate to the `harness` directory and run the `labeler.ts` script.

```bash
cd harness
npm install

# Example: Run the labeler on the HLE dataset using Gemini
npx tsx labeler.ts --dataset hle --model gemini-3.5-flash

# Use Anthropic instead (provider is also inferred from a claude-* model name)
npx tsx labeler.ts --dataset strong_reject --provider anthropic --model claude-opus-5-5

# Re-label a dataset from scratch (archives existing labels + human patches to labels/archive/)
npx tsx labeler.ts --dataset strong_reject --provider anthropic --relabel
```
*(See `harness/run_evals.sh` for batch-running examples).*

API keys are read from `tools/label/.env`: `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`. Both providers use JSON-schema structured output: Gemini's `responseSchema` and Anthropic's `output_config.format`. The schema's enums come from `tools/label/taxonomy.csv` (its `subcategory` rows are reporting-only and excluded), so responses always parse and only contain valid label ids. Other flags: `--limit N` labels only the first N new samples, and `--mock` makes no API calls.

**Refusals:** red-teaming prompts sometimes trip the provider's own safety filter. When Claude refuses and `GEMINI_API_KEY` is set, the sample is retried with Gemini automatically, and the row's `llm_assigned.model` records which model labeled it. Samples that are still refused are logged as `🚫 REFUSED` and saved with low confidence, so the UI marks them for review. Re-running the same command retries them; anything that keeps getting refused should be labeled by hand in the UI.


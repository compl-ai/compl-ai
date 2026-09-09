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
```
*(See `harness/run_evals.sh` for batch-running examples).*

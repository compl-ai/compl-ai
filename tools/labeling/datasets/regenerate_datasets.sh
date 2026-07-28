#!/bin/bash

# Simple script to regenerate original datasets locally using the compl-ai CLI.
# This prevents checking in heavy, licensed datasets to git.

cd ../../.. # Go to project root to run UV

# Get all available datasets dynamically from complai tasks
DATASETS=($(uv run python -c 'from complai._cli.utils import get_complai_tasks; print(" ".join(t.name for t in get_complai_tasks()))'))

echo "Regenerating all datasets..."

for DATASET in "${DATASETS[@]}"; do
    echo "Regenerating: $DATASET"
    # Using inspect_evals task naming convention or direct compl-ai tasks
    # (Update task name mapping if necessary for specific datasets)
    uv run complai samples "tools/labeling/datasets/${DATASET}.jsonl" -t "$DATASET" || echo "Failed to generate $DATASET (task name might differ)"
done

echo "Done!"

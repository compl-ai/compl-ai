#!/bin/bash
set -e
models=("gemini-3.1-pro-preview" "gemini-3.6-flash" "gemini-3.5-flash" "gemini-3.5-flash-lite" "gemini-3.1-flash-lite")
dataset="bigbench_calibration"
limit=80

for model in "${models[@]}"; do
    echo "Running $model on $dataset..."
    npx tsx labeler.ts --dataset $dataset --limit $limit --model $model
done

#!/bin/bash

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "uv could not be found. Install from https://docs.astral.sh/uv/getting-started/installation"
    exit 1
fi


tasks=compl-ai
model=hf
model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct"
limit=1
log_samples=true

# HumanEval requires confirmation to run unsafe code
confirm_run_unsafe_code=true
export HF_ALLOW_CODE_EVAL=1


uv run -- lm-eval \
    --tasks $tasks \
    --model $model \
    --model_args "$model_args" \
    --device cuda \
    --include_path src/complai/tasks \
    --limit $limit \
    --output_path logs \
    ${log_samples:+--log_samples} \
    ${confirm_run_unsafe_code:+--confirm_run_unsafe_code}
#!/bin/bash

model=meta-llama/Llama-3.1-8B-Instruct


# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "uv could not be found. Please install it from https://docs.astral.sh/uv"
    exit 1
fi


# Ask for confirmation to run HumanEval
if [ -z "$confirm_run_unsafe_code" ]; then
    read -p "HumanEval requires permission to run potentially unsafe code. Do you want to set confirm_run_unsafe_code=True to run this task? (y/n): " ans
    if [ "$ans" != "y" ]; then
        # for now we just stop, could just run without HumanEval
        exit 1
    else
        confirm_run_unsafe_code=true
        export HF_ALLOW_CODE_EVAL=1
    fi
fi


uv run -- lm-eval \
    --tasks compl-ai \
    --include_path src/complai/tasks \
    --model hf \
    --model_args pretrained=$model \
    --device cuda \
    --log_samples \
    --output_path logs \
    ${confirm_run_unsafe_code:+--confirm_run_unsafe_code} \
    --limit 2
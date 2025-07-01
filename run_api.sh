#!/bin/bash

MODEL=swissai/apertus3-70b-5T-sft
BASE_URL=https://api.swissai.cscs.ch/v1/chat/completions
TASKS=compl-ai # Can be a subset of tasks/groups, e.g., "TASKS=human_deception,capabilities_performance_and_limitations"
OUTPUT_PATH="logs"
LIMIT=1

LOG_SAMPLES=true
APPLY_CHAT_TEMPLATE=true
FEWSHOT_AS_MULTITURN=true
MODEL_ARGS="model=$MODEL,base_url=$BASE_URL,max_retries=2,tokenized_requests=False,num_concurrent=16"

# HumanEval requires confirmation to run unsafe code
CONFIRM_RUN_UNSAFE_CODE=true
export HF_ALLOW_CODE_EVAL=1

# Functions
check_dependencies() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv not found. Install from https://docs.astral.sh/uv/getting-started/installation" >&2
        exit 1
    fi
}

check_api_key() {
    if [[ -z "${COMPLAI_API_KEY:-}" ]]; then
        echo "Error: COMPLAI_API_KEY environment variable is not set." >&2
        exit 1
    fi
}

build_lm_eval_args() {
    local args=(
        --tasks "$TASKS"
        --model local-chat-completions
        --model_args "$MODEL_ARGS"
        --include_path src/complai/tasks
        --output_path "$OUTPUT_PATH"
    )
    
    # Add conditional arguments
    [[ "$APPLY_CHAT_TEMPLATE" == true ]] && args+=(--apply_chat_template)
    [[ "$FEWSHOT_AS_MULTITURN" == true ]] && args+=(--fewshot_as_multiturn)
    [[ "$LOG_SAMPLES" == true ]] && args+=(--log_samples)
    [[ "$CONFIRM_RUN_UNSAFE_CODE" == true ]] && args+=(--confirm_run_unsafe_code)
    [[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")
    
    echo "${args[@]}"
}

run_evaluation() {
    echo "Running evaluation with model: $MODEL"
    echo "Tasks: $TASKS"
    echo "Output path: $OUTPUT_PATH"
    echo
    
    # lm-eval expects OPENAI_API_KEY environment variable
    export OPENAI_API_KEY="$COMPLAI_API_KEY"
    
    local args
    read -ra args <<< "$(build_lm_eval_args)"
    
    uv run -- lm-eval "${args[@]}"
}

check_dependencies
check_api_key
run_evaluation

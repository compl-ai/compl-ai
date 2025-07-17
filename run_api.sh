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

check_and_filter_logprobs_tasks() {
    # Check if task list contains any tasks that require logprobs
    # lm-eval API models do not support returning logprobs, so we remove them from the task list.
    # This function returns the filtered task list.
    local tasks="$1"
    local logprobs_tasks=("bigbench_calibration" "redditbias")
    local temp_file=$(mktemp)
    local result_file=$(mktemp)
    
    echo "You are evaluating an API model. API models do not support tasks that require logprobs." >&2
    echo "Checking provided tasks for logprobs-requiring tasks..." >&2
    echo >&2
    uv run -- python3 -c "
from lm_eval.tasks import TaskManager
from lm_eval.evaluator_utils import get_subtask_list

def get_leaf_tasks(task_dict):
    '''Recursively get all leaf tasks (actual tasks, not groups)'''
    leaf_tasks = set()
    
    def extract_tasks(obj):
        if hasattr(obj, 'config') and hasattr(obj.config, 'task'):
            # This is a ConfigurableTask - a leaf task
            leaf_tasks.add(obj.config.task)
        elif isinstance(obj, dict):
            # This is a nested dictionary, recurse
            for value in obj.values():
                extract_tasks(value)
        elif hasattr(obj, '__iter__') and not isinstance(obj, str):
            # This is some other iterable, recurse
            for item in obj:
                extract_tasks(item)
    
    extract_tasks(task_dict)
    return sorted(list(leaf_tasks))

task_manager = TaskManager(include_path='src/complai/tasks', verbosity='CRITICAL')
task_list = '$tasks'.split(',')

task_dict = task_manager.load_task_or_group(task_list)
leaf_tasks = get_leaf_tasks(task_dict)

with open('$temp_file', 'w') as f:
    for task_name in leaf_tasks:
        f.write(task_name + '\n')
" >/dev/null 2>&1
    
    local expanded_tasks
    expanded_tasks=$(cat "$temp_file")
    rm -f "$temp_file"
        
    local found_tasks=()
    for task in "${logprobs_tasks[@]}"; do
        if echo "$expanded_tasks" | grep -q "^$task$"; then
            found_tasks+=("$task")
        fi
    done
    
    if [ ${#found_tasks[@]} -gt 0 ]; then
        echo "----------------------------- WARNING -----------------------------" >&2
        echo "The following tasks require logprobs which are not supported by API models:" >&2
        printf "  - %s\n" "${found_tasks[@]}" >&2
        echo "These tasks will be excluded from the evaluation." >&2
        echo >&2
        
        local filtered_tasks=""
        while IFS= read -r task; do
            local skip=false
            for problem_task in "${found_tasks[@]}"; do
                if [ "$task" = "$problem_task" ]; then
                    skip=true
                    break
                fi
            done
            if [ "$skip" = false ]; then
                if [ -z "$filtered_tasks" ]; then
                    filtered_tasks="$task"
                else
                    filtered_tasks="$filtered_tasks,$task"
                fi
            fi
        done <<< "$expanded_tasks"
        
        if [ -z "$filtered_tasks" ]; then
            rm -f "$result_file"
            exit 1
        fi
        
        echo "Filtered task list: $filtered_tasks" >&2
        echo "$filtered_tasks" > "$result_file"
    else
        echo "$tasks" > "$result_file"
    fi
    
    cat "$result_file"
    rm -f "$result_file"
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
    echo "Model: $MODEL"
    echo "API URL: $BASE_URL"
    echo "Tasks: $TASKS"
    echo "Output path: $OUTPUT_PATH"
    echo
    
    # Check for logprobs-requiring tasks and remove if needed
    TASKS=$(check_and_filter_logprobs_tasks "$TASKS")
    if [ -z "$TASKS" ]; then
        echo "No tasks remaining after excluding tasks requiring logprobs. Exiting." >&2
        exit 1
    fi
    
    # lm-eval expects OPENAI_API_KEY environment variable
    export OPENAI_API_KEY="$COMPLAI_API_KEY"
    
    local args
    read -ra args <<< "$(build_lm_eval_args)"
    
    uv run -- lm-eval "${args[@]}"
}

check_dependencies
check_api_key
run_evaluation

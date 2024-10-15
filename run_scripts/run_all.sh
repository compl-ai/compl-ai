#!/bin/bash
cd ..
RUN_NAME="gpt-neo-125m"
MODEL_PATH="EleutherAI/gpt-neo-125m" 
DEBUG_MODE="" # "--debug_mode"
CURRENT_DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")

benchmarks_req_open_ai_keys=("self_check_consistency")
if [ -z "${OPENAI_API_KEY}" ] | [ -z "${OPENAI_ORG}" ]; then
  echo -e "[COMPL-AI] Warning: 'OPENAI_API_KEY' and 'OPENAI_ORG' variables are not set."
  echo -e "[COMPL-AI] Please set them to avoid skipping the following benchmarks that use it:"

  # Loop through the list and print each string
  for item in "${benchmarks_req_open_ai_keys[@]}"; do
    echo "[COMPL-AI]   - $item"
  done
fi

run_job () {
  batch_size=${2:-10}
  model_config=${3:-"configs/models/default_model.yaml"}
  answers_file=${4:-""}
        poetry run python3 run.py $DEBUG_MODE \
        --model_config=$model_config \
        --model=$MODEL_PATH \
        --batch_size=$batch_size \
        --results_folder="runs/$RUN_NAME/$CURRENT_DATETIME" \
        --answers_file=$answers_file \
        $1 \
	> "runs/$RUN_NAME/$CURRENT_DATETIME/$(echo $1 | sed 's#/#_#g').log" \
	2> "runs/$RUN_NAME/$CURRENT_DATETIME/$(echo $1 | sed 's#/#_#g').errors"
}


mkdir -p runs/$RUN_NAME/$CURRENT_DATETIME



# Capabilities
run_job configs/capabilities/human_eval.yaml 10 configs/models/human_eval_model.yaml
run_job configs/capabilities/hellaswag.yaml
run_job configs/capabilities/ai2_reasoning.yaml
run_job configs/capabilities/truthful_qa_mc2.yaml
run_job configs/capabilities/mmlu.yaml 

# Cybersecurity
run_job configs/cybersecurity/instruction_goal_hijacking.yaml 1 # Not yet optimized for batches
run_job configs/cybersecurity/multiturn_goal_hijacking.yaml 1 # Not yet optimized for batches

# Calibration
run_job configs/calibration/calibration_big_bench_i_know.yaml
run_job configs/calibration/calibration_big_bench.yaml

# Fairness
run_job configs/fairness/bold.yaml
run_job configs/fairness/bbq.yaml
run_job configs/fairness/fairllm.yaml
run_job configs/fairness/decoding_trust.yaml 1

# Bias
run_job configs/bias/reddit_bias.yaml 1 # Not yet optimized for batches


# Robustness & Predictability
run_job configs/consistency/self_check_consistency.yaml 
run_job configs/consistency/forecasting_consistency.yaml
run_job configs/robustness/boolq_contrast_robustness.yaml
run_job configs/robustness/imdb_contrast_robustness.yaml

DEBUG_MODE="--debug_mode"
run_job configs/robustness/mmlu_robustness.yaml
# Toxicity
run_job configs/toxicity/toxicity.yaml
run_job configs/toxicity/toxicity_advbench.yaml
DEBUG_MODE=""
# Privacy
run_job configs/privacy/privacy.yaml

# Copyright 
run_job configs/memorization/memorization.yaml

# Disclosure of AI Presence
run_job configs/human_deception/human_deception.yaml 1 # Not yet optimized for batches

poetry run python3 helper_tools/results_processor.py --parent_dir=runs/$RUN_NAME/$CURRENT_DATETIME --model_name=$MODEL_PATH

#!/bin/bash
cd .. 
RUN_NAME="gpt-4-1106-preview"
MODEL_PATH="gpt-4-1106-preview"
DEBUG_MODE="" # "--debug_mode"
CURRENT_DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")

run_job () {
  batch_size=${2:-10}
  model_config=${3:-"configs/models/openai_model.yaml"}
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
DEBUG_MODE=""
# Privacy
run_job configs/privacy/privacy.yaml

# Copyright 
run_job configs/memorization/memorization.yaml

# Disclosure of AI Presence
run_job configs/human_deception/human_deception.yaml 1 # Not yet optimized for batches

poetry run python3 helper_tools/results_processor.py --parent_dir=runs/$RUN_NAME/$CURRENT_DATETIME --model_name=$MODEL_PATH

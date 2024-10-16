#!/bin/bash

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



echo "Principle: Transparency"

# Capabilities
echo "└── Requirement: Capabilities, Performance, and Limitations"
echo "    ├── Coding: HumanEval..."
run_job configs/capabilities/human_eval.yaml 10 configs/models/human_eval_model.yaml
echo "    ├── Coding: Common Sense Reasoning: HellaSwag..."
run_job configs/capabilities/hellaswag.yaml
echo "    ├── Reasoning: AI2 Reasoning Challenge..."
run_job configs/capabilities/ai2_reasoning.yaml
echo "    ├── Truthfulness: TruthfulQA MC2..."
run_job configs/capabilities/truthful_qa_mc2.yaml
echo "    └── General Knowledge: MMLU..."
run_job configs/capabilities/mmlu.yaml

# Calibration
echo "└── Requirement: Interpretability"
echo "    ├── Self-Assessment: TriviaQA..."
run_job configs/calibration/calibration_big_bench_i_know.yaml
echo "    └── Logit Calibration: BIG-Bench..."
run_job configs/calibration/calibration_big_bench.yaml

# Disclosure of AI Presence
echo "└── Requirement: Disclosure of AI"
echo "    └── Denying Human Presence..."
run_job configs/human_deception/human_deception.yaml 1 # Not yet optimized for batches

# Traceability
echo "└── Requirement: Traceability"
echo "    └── Watermark Reliability & Robustness... [Unavailable]"


echo "Principle: Technical Robustness and Safety"

# Cybersecurity
echo "└── Requirement: Cyberattack Resilience"
echo "    ├── Goal Hijacking and Prompt Leakage: TensorTrust..."
run_job configs/cybersecurity/instruction_goal_hijacking.yaml 1 # Not yet optimized for batches
echo "    └── Rule Following: LLM RuLES..."
run_job configs/cybersecurity/multiturn_goal_hijacking.yaml 1 # Not yet optimized for batches

# Robustness & Predictability
echo "└── Requirement: Robustness and Predictability"
echo "    ├── Self-Check Consistency..."
run_job configs/consistency/self_check_consistency.yaml
echo "    ├── Motononicity Checks..."
run_job configs/consistency/forecasting_consistency.yaml
echo "    ├── BoolQ Contrast Set..."
run_job configs/robustness/boolq_contrast_robustness.yaml
echo "    ├── IMDB Contrast Set..."
run_job configs/robustness/imdb_contrast_robustness.yaml
DEBUG_MODE="--debug_mode"
echo "    └── MMLU: Robustness..."
run_job configs/robustness/mmlu_robustness.yaml
DEBUG_MODE=""

echo "Principle: Diversity, Non-discrimination & Fairness"
# Bias
echo "└── Requirement: Representation — Absence of Bias"
echo "    ├── Representation Bias: RedditBias..."
run_job configs/bias/reddit_bias.yaml 1 # Not yet optimized for batches
echo "    ├── Biased Completions: BOLD..."
run_job configs/fairness/bold.yaml
echo "    └── Prejudiced Answers: BBQ..."
run_job configs/fairness/bbq.yaml

# Fairness
echo "└── Requirement: Fairness — Absence of Discrimination"
echo "    ├── Recommendation Consistency: FaiRLLM..."
run_job configs/fairness/fairllm.yaml
echo "    └── Income Fairness: DecodingTrust..."
run_job configs/fairness/decoding_trust.yaml 1



echo "Principle: Social & Environmental Well-being"
echo "└── Requirement: Harmful Content and Toxicity"
DEBUG_MODE="--debug_mode"
# Toxicity
echo "    ├── Toxic Completions of Benign Text: RealToxicityPrompts..."
run_job configs/toxicity/toxicity.yaml
echo "    └── Following Harmful Instructions: AdvBench..."
run_job configs/toxicity/toxicity_advbench.yaml
DEBUG_MODE=""

echo "Principle: Privacy & Data Governance"
# Privacy
echo "└── Requirement: User Privacy Protection"
echo "    └── PII Extraction by Association..."
run_job configs/privacy/privacy.yaml

# Copyright
echo "└── Requirement: No Copyright Infringement "
echo "    └── Copyrighted Material Memorization..."
run_job configs/memorization/memorization.yaml

echo "└── Requirement: Training Data Suitability "
echo "    ├── Toxicity of the Dataset... [Unavailable]"
echo "    └── Bias of the Dataset... [Unavailable]"


poetry run python3 helper_tools/results_processor.py --parent_dir=runs/$RUN_NAME/$CURRENT_DATETIME --model_name=$MODEL_PATH

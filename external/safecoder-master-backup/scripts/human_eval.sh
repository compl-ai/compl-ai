#!/bin/bash

OUTPUT_NAME=${1}
MODEL_NAME=${2}

python human_eval_gen.py --output_name ${OUTPUT_NAME} --model_name ${MODEL_NAME}
python human_eval_exec.py --output_name ${OUTPUT_NAME}
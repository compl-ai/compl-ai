## Setup
Set up Python dependencies (a virtual environment is recommended) and [GitHub CodeQL](https://github.com/github/codeql):
```console
pip install -r requirements.txt
pip install -e .
./setup_codeql.sh
```
## Usages
Evaluating pretrained LMs:
```console
python sec_eval.py --output_name lm-1b --model_name starcoderbase-1b
./human_eval.sh lm-1b starcoderbase-1b
```

Run instruction-tuning for functional correctness:
```console
python train.py --output_name func-1b --datasets code-alpaca --pretrain_name starcoderbase-1b
python sec_eval.py --output_name func-1b --model_name func-1b
./human_eval.sh func-1b func-1b
```

Run instruction-tuning for functional correctness and security:
```console
python train.py --output_name sec-1b --datasets sec code-alpaca --pretrain_name starcoderbase-1b
python sec_eval.py --output_name sec-1b --model_name sec-1b
./human_eval.sh sec-1b sec-1b
```
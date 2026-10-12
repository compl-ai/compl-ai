<p align="center">
  <a href="https://compl-ai.org"><img  style="max-height: 50px;" src="compl-ai-logo.svg" alt="COMPL-AI">
</a>
</p>

<div align="center">
    COMPL-AI is a compliance-centered evaluation framework for LLMs created and maintained by
      
  [ETH Zurich](https://www.sri.inf.ethz.ch/), [INSAIT](https://insait.ai/) and [LatticeFlow AI](https://latticeflow.ai/).
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.07959-b31b1b)](https://arxiv.org/abs/2410.07959)
[![Web](https://img.shields.io/badge/Website-compl--ai.org-blue)](https://compl-ai.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-g)](LICENSE)

</div>

## Overview
The COMPL-AI framework includes a [technical interpretation](https://compl-ai.org/interpretation/) of the EU AI Act and an [open-source benchmarking suite](https://github.com/compl-ai/compl-ai/) (this repo). The key features are:

- Built on the [Inspect evaluation framework](https://github.com/UKGovernmentBEIS/inspect_ai) 
- Tailored set of benchmarks to provide coverage over technical parts of EU AI Act (30 and growing)
- A public Hugging Face leaderboard of our [latest evaluation results](https://huggingface.co/spaces/latticeflow/compl-ai-board)   
- Extensive set of supported [providers](providers/README.md) (API, Cloud, Local).
- A custom eval CLI (run `complai --help` for usage)

Community contributions for benchmarks and new mappings are welcome! We are actively looking to expand our EU AI Act and Code of Practice technical interpretation and benchmark coverage. See the [contributing](#-contributing) section below.

## ⏩ Quickstart
To run an evaluation yourself, please follow the instructions below (or contact us through [compl-ai.org](https://compl-ai.org)). 


```bash
# Clone and create a virtual environment
git clone https://github.com/compl-ai/compl-ai.git
cd compl-ai
uv sync
source .venv/bin/activate

# Set your API keys (HF_TOKEN needs access to the gated datasets listed under Checks and Reports)
export OPENAI_API_KEY=your_key
export HF_TOKEN=your_token

# Check that every task runs, on 1 item per task (smoke runs are never used for scores)
complai eval openai/gpt-5-nano --smoke 1

# Run the core subset (2,000 items)
complai eval openai/gpt-5-nano
```

You can then view a detailed sample-level log of your results with the [Inspect AI VS Code extension](https://marketplace.visualstudio.com/items?itemName=ukaisi.inspect-ai), or in your browser with:
```
inspect view
```


## 💻 CLI


#### Get help with any command
```bash
complai COMMAND --help
```

#### List Tasks

```bash
complai list
```


#### Run Evals with the following syntax
```bash
complai eval <provider>/<model> [-t <task_name>] [--smoke N | --full]
```

By default `complai eval` runs the COMPL-AI Core subset (see below). Before each run it checks that the model, the judge models and the gated datasets are reachable, and after each run it prints a per-task report.

#### Command Examples
```bash
# Core subset, all tasks or one task at a time
complai eval openai/gpt-4o-mini
complai eval anthropic/claude-sonnet-4-0 -t mmlu_pro

# Smoke test: first 2 items of each core task, to check everything runs (never used for scores)
complai eval openai/gpt-4o-mini --smoke 2

# Full suite; --limit, --sample-id and --task-filter only work with --full
complai eval openai/gpt-4o-mini --full
complai eval openai/gpt-4o-mini --full -t mmlu_pro --limit 5

# Locally with HF backend, set cuda device (use mps for macOS)
complai eval hf/Qwen/Qwen3-8B -t mmlu_pro -M device=cuda:0

# Using vLLM backend, evaluate specific sample and cap number of sandboxes for agentic benchmarks
complai eval vllm/Qwen/Qwen3-8B --full -t swe_bench_verified --sample-id django__django-11848 --max-sandboxes 1 

# Use task configuration file or CLI task args (CLI args take precedence)
complai eval openai/gpt-5-nano --task-config config/default_config.yaml -T mmlu_pro:num_fewshot=5

# Retry (if eval failed) with existing log directory or specify custom log directory (supports S3 URLs)
complai eval openai/gpt-5-nano --log-dir path/to/logdir
```

#### Checks and Reports
```bash
# Run the pre-eval checks on their own
complai status openai/gpt-4o-mini

# Per-task status, errors, limits, truncated outputs, coverage and metrics for a run
complai report logs/<folder_name>
```

Some tasks are graded by OpenAI judge models, so `OPENAI_API_KEY` is needed even when the evaluated model is not from OpenAI. If it is not set but `OPENROUTER_API_KEY` is, the same judges are used through OpenRouter. Judges set explicitly with `-T` are left unchanged.

Two tasks load gated Hugging Face datasets: `hle` ([cais/hle](https://huggingface.co/datasets/cais/hle)) and `strong_reject`, whose jailbreak prompts come pre-generated from [compl-ai/strong_reject](https://huggingface.co/datasets/compl-ai/strong_reject) so every run uses the same prompts as the reference runs. Request access on both pages and set `HF_TOKEN` (or run `hf auth login`).

#### Task Configuration
COMPL-AI supports task-specific configuration via YAML or JSON files. See `config/default_config.yaml` for a reference of all configurable parameters (regenerate it with `make default-config`). You can:

- Use the default config as a template: `cp config/default_config.yaml my_config.yaml`
- Modify the tasks and parameters you want
- Pass it to any eval: `complai eval <model> --task-config my_config.yaml`

#### Providers
See the [Providers](providers/README.md) section for more information on different providers.

#### Environment Variables
COMPL-AI can auto-load models (`COMPLAI_MODEL`), API keys (`OPENAI_API_KEY`), and many other configurations (`COMPLAI_LOG_DIR`) from your local `.env` file. Values provided in the CLI take precedence over `.env` vars.


#### COMPL-AI Core (Subset Evaluation)

COMPL-AI Core is a fixed set of 2,000 highly discriminative items chosen from the full benchmark suite. From a model's results on these items, an IRT model predicts its capability, reliability and safety scores on the full suite.

`complai eval` runs the bundled Core subset by default:

```bash
# 1. Run the evaluation on the 2,000 items
complai eval openai/gpt-5-nano --log-dir logs/

# 2. Predict the full-suite scores
complai predict logs/<folder_name> --output predicted.json
```
*(Maintainers: See `tools/minify/README.md` for instructions on regenerating the Core subset).*

#### Collect Sample Inputs

Export samples as JSONL:

```bash
complai samples samples.jsonl
complai samples samples.jsonl --tasks mmlu_pro,human_deception --limit 10
complai samples arc.jsonl --task-spec inspect_evals/arc_challenge
```

Each record contains the task name, sample ID, and the structured input that
the task dataset supplies to the evaluation pipeline.

## 🧪 Framework

#### 🚧 Update in Progress
The current version of the framework is published [here](https://compl-ai.org/interpretation/). 

We are currently in the process of renewing our coverage of the EU AI Act by updating the set of benchmarks, thus the supported set of benchmark may differ from this original mapping. The goals of this update are:
- To increase coverage over the EU AI Act principles
- To increase coverage over technical requirements
- Adding support for the Code of Practice, namely the [Safety and Security](https://code-of-practice.ai/?section=safety-security) chapter.
- Adding the notion of `risk` along side `technical requirements`
- Refreshing the supported benchmarks to ensure they remain challenging for frontier models (addressing saturation, contamination, and other benchmark quality issues).

As part of our update, the renewed benchmarking suite is now built on the UK AI Security Institute’s [Inspect Framework](https://github.com/UKGovernmentBEIS/inspect_ai), which offers improved ease of use and greater overall consistency. This means that several benchmarks are now evaluated differently (e.g. full-text answers instead of logits for multiple-choice questions) reflecting more modern and opinionated approaches to LLM evaluation. Thus, benchmark scores from the v1 and v2 suites, even for the same benchmark, should not be considered directly comparable.

### Principles
COMPL-AI is primarily structured to provide coverage over 6 core EU AI Act principles:
- Human Agency and Oversight: AI systems should be supervised by people, not by automation alone, to prevent harmful outcomes and allow for human intervention. 
- Technical Robustness and Safety: AI systems must be safe and secure, implementing risk management, data quality, and cybersecurity measures to prevent undue risks. 
- Privacy and Data Governance: The Act sets rules for the quality and governance of data used in AI, emphasizing the protection of personal and sensitive information. 
- Transparency: Users should understand when they are interacting with an AI system and how it functions, fostering trust and enabling accountability. 
- Diversity, Non-Discrimination, and Fairness: AI systems should be designed and used to uphold human rights, including fairness and equality, and avoid biases that could lead to discrimination. 
- Societal and Environmental Well-being: AI systems should be developed in a way that benefits society and the environment, avoiding negative impacts on fundamental rights and democratic values. 

### Technical Requirements and Benchmarks
You can see a list of all technical requirements and their respective benchmarks using `complai list`:
- Capabilities, Performance, and Limitations
  - aime_2025, arc_challenge, gpqa_diamond, hle, ifbench, include, livebench_coding, mmlu_pro, mmmu_pro, swe_bench_verified
- Representation — Absence of Bias
  - bbq, bold, cab
- Interpretability
  - bigbench_calibration, triviaqa_calibration
- Robustness and Predictability
  - boolq_contrast, forecast_consistency, imdb_contrast, mmlu_pro_robustness, self_check_consistency
- Fairness — Absence of Discrimination
  - decoding_trust, fairllm
- Disclosure of AI
  - human_deception
- Cyberattack Resilience
  - instruction_goal_hijacking, llm_rules, strong_reject
- Societal Alignment
  - mask, simpleqa_verified, truthfulqa
- Harmful Content and Toxicity
  - realtoxicityprompts



## 🤝 Contributing
We welcome contributions! When contributing, please make sure to activate pre-commit hooks to ensure code quality and consistency. You can install pre-commit hooks with:

```bash
pip install pre-commit
pre-commit install
```

To run tests, run the following command:

```bash
make test
```


## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 📝 Citation

Please cite our work as follows:

```
@article{complai24,
      title={COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act}, 
      author={Philipp Guldimann and Alexander Spiridonov and Robin Staab and Nikola Jovanovi\'{c} and Mark Vero and Velko Vechev and Anna Gueorguieva and Mislav Balunovi\'{c} and Nikola Konstantinov and Pavol Bielik and Petar Tsankov and Martin Vechev},
      year={2024},
      eprint={2410.07959},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.07959},
}
```

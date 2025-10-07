<p align="center">
  <a href="https://compl-ai.org"><img  style="max-height: 50px;" src="https://compl-ai.org/compl-ai-logo.svg" alt="COMPL-AI">
</a>
</p>

<div align="center">
    COMPL-AI is a compliance-centered evaluation framework for Generative AI models, created and maintained by
      
  [ETH Zurich](https://www.sri.inf.ethz.ch/), [INSAIT](https://insait.ai/) and [LatticeFlow AI](https://latticeflow.ai/).
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.07959-b31b1b)](https://arxiv.org/abs/2410.07959)
[![Web](https://img.shields.io/badge/Website-compl--ai.org-blue)](https://compl-ai.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-g)](LICENSE)

</div>

## Overview
COMPL-AI is an EU AI Act benchmarking framework allowing for technical assessment of LLMs. It includes a [technical interpretation](https://compl-ai.org/interpretation/) of the EU AI Act and an [open-source benchmarking suite](https://github.com/compl-ai/compl-ai/) (this repo). The key features are:

- Built on the [Inspect evaluation framework](https://github.com/UKGovernmentBEIS/inspect_ai) 
- Tailored set of benchmarks to provide coverage over technical parts of EU AI Act (23 and growing)
- A public Hugging Face leaderboard of our [latest evaluation results](https://huggingface.co/spaces/latticeflow/compl-ai-board)   
- Extensive set of supported [providers](providers/README.md) (API, Cloud, Local).
- A custom eval CLI (run `complai --help` for usage)

Community contributions for benchmarks and new mappings are welcome! We are actively looking to exapand our EU AI Act and Code of Practice technical interpretation and benchmark coverage. See the [contributing](#-contributing) section below.

## ⏩ Quickstart
To run an evaluation yourself, please follow the instructions below (or contact us through [compl-ai.org](https://compl-ai.org)). 


```bash
# Clone and create a virtual environment
git clone https://github.com/compl-ai/compl-ai.git
cd compl-ai
uv venv
source .venv/bin/activate

# Set your API key
export OPENAI_API_KEY=your_key

# Run 5 samples on a single benchmark
complai eval openai/gpt-5-nano --tasks mmlu_pro --limit 5

# Or run the full framework
complai eval openai/gpt-5-nano
```

You can then view a detailed sample-level log of your results with the [Inspect AI VS Code extension](https://marketplace.cursorapi.com/items/?itemName=ukaisi.inspect-ai), or in your browser with:
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
complai eval <provider>/<model> -t <task_name> -l <n_samples>
```

#### Command Examples
```bash
# Remote API
complai eval openai/gpt-4o-mini
complai eval anthropic/claude-sonnet-4-0

# Locally with HF backend, set cuda device (use mps for macOS)
complai eval hf/Qwen/Qwen3-8B -t mmlu_pro -M device=cuda:0

# Using vLLM backend, target specific sample and cap number of sandboxes for agentic benchmarks
complai eval vllm/Qwen/Qwen3-8B -t swe_bench_verified --sample-id django__django-11848 --max-sandboxes 1 

# Retry (if eval failed)
complai eval openai/gpt-5-nano --log-dir path/to/logdir
```

See the [Providers](providers/README.md) section for more information on different providers.

#### `Env` Vars
COMPL-AI will auto-load models `COMPLAI_MODEL`, api keys `OPENAI_API_KEY`, and other vars  `COMPLAI_LOG_DIR` from your local `env` file. The CLI values take precedence over `.env` vars.


## 🧪 Framework

#### 🚧 Update in Progress
The current version of the framework is published here [here](https://compl-ai.org/interpretation/). 

We are currently in the process of renewing our coverage of the EU AI Act by updating the set of benchmarks, thus the supported set of benchmark may differ from this original mapping. The goals of this update are:
- To increase coverage over the EU AI Act principles
- To increase coverage over technical requirements
- Adding support for the Code of Practice, namely the [Safety and Security](https://code-of-practice.ai/?section=safety-security) chapter.
- Adding the notion of `risk` along side `technical requirements`
- Refreshing the supported benchmarks to ensure they remain challenging for frontier models (addressing saturation, contamination, and other benchmark quality issues).

### Principles
COMPL-AI is primarily structured to provide coverage over 6 core EU AI Act priciples:
- Human Agency and Oversight: AI systems should be supervised by people, not by automation alone, to prevent harmful outcomes and allow for human intervention. 
- Technical Robustness and Safety: AI systems must be safe and secure, implementing risk management, data quality, and - cybersecurity measures to prevent undue risks. 
- Privacy and Data Governance: The Act sets rules for the quality and governance of data used in AI, emphasizing the protection of personal and sensitive information. 
- Transparency: Users should understand when they are interacting with an AI system and how it functions, fostering trust and enabling accountability. 
- Diversity, Non-Discrimination, and Fairness: AI systems should be designed and used to uphold human rights, including fairness and equality, and avoid biases that could lead to discrimination. 
- Societal and Environmental Well-being: AI systems should be developed in a way that benefits society and the environment, avoiding negative impacts on fundamental rights and democratic values. 

### Technical Requirements and Benchmarks
You can see list of all technical requirements and their respective benchmarks using `complai list`:
- Capabilities, Performance, and Limitations
  - aime_2025, arc_challenge, hellaswag, humaneval, livebench_coding, mmlu_pro, swe_bench_verified, truthfulqa
- Representation — Absence of Bias
  - bbq, bold
- Interpretability
  - bigbench_calibration, triviaqa_calibration
- Robustness and Predictability
  - boolq_contrast, forecast_consistency, imdb_contrast, self_check_consistency
- Fairness — Absence of Discrimination
  - decoding_trust, fairllm
- Disclosure of AI
  - human_deception
- Cyberattack Resilience
  - instruction_goal_hijacking, llm_rules
- Harmful Content and Toxicity
  - realtoxicityprompts




## 🤝 Contributing
We welcome contributions! When contributing, please make sure to activate pre-commit hooks to ensure code quality and consistency. You can install pre-commit hooks with:

```
pip install pre-commit
pre-commit install
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
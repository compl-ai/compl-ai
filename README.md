<div>
  <p>
    <a href="https://compl-ai.org" target="_blank">
      <img width="100%" style="max-width: 600px;" src="banner.png" alt="YOLO Vision banner"></a>
  </p>
</div>


This repository contains the open-source framework and the corresponding technical mapping for evaluating generative AI models.
- To run the evaluation yourself, please follow the instructions below.
- To request an evaluation, please contact us through the [compl-ai.org](https://compl-ai.org) website. 

This project created by [ETH Zurich](https://www.sri.inf.ethz.ch/), [INSAIT](https://insait.ai/) and [LatticeFlow AI](https://latticeflow.ai/).

## Installation

Prerequisite: [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

### 1. Clone this repository
```
git clone https://github.com/compl-ai/compl-ai.git
cd compl-ai
```
### 2a. Using `uv run`

You can run the CLI directly using `uv run`:

```
uv run complai --help
```

### 2b. Manual Installation

You can also create a virtual environment and install the package manually. Using `uv`:

```
uv venv
source .venv/bin/activate
uv pip install .
```
Now you can use the CLI like so:

```
complai --help
```


## Quickstart

The command 
```
uv run complai eval provider/model
```
runs *all* benchmarks. Concrete examples:

```
# OpenAI
export OPENAI_API_KEY=your-openai-api-key
uv run complai eval openai/gpt-5-nano --limit 5
```
```
# Local (Starts a vLLM server)
uv run complai eval vllm/HuggingFaceTB/SmolLM2-135M-Instruct --limit 5 
```
```
# OpenAI-compatible API endpoint
EXPORT {provider_name}_API_KEY
uv run complai eval openai-api/{provider_name}/{model_name} --base-url https://your.base/url --limit 5
```
You can view detailed sample-level logs with the [Inspect AI VS Code extension](https://marketplace.cursorapi.com/items/?itemName=ukaisi.inspect-ai), or in your browser with:
```
uv run inspect view
```


## CLI


```
complai COMMAND [ARGS]...
```

**Available Commands**:

* `eval`: Run tasks.
* `list`: List all available tasks.

Run `complai COMMAND --help` for information on CLI arguments and options.

### Retry

To continue interrupted tasks or retry failed tasks, you can specify a `--log-dir` argument in the `eval` command. This will automatically retry the tasks in the log directory.
```
uv run complai eval --log-dir path/to/logdir
```

### Environment Variables

The CLI supports reading argument and option values from environment variables. For instance, you can run:
```
export COMPLAI_MODEL=openai/gpt-5-nano
```
 This model will then be used if no model is provided to the `eval` command.

You can also use a `.env` file to set environment variables:
```
# In your .env file add:
OPENAI_API_KEY=your-openai-api-key

COMPLAI_MODEL=openai/gpt-5-nano
COMPLAI_LOG_DIR=path/to/a/logdir
COMPLAI_MAX_CONNECTIONS=128
```
COMPL-AI will automatically load variables from a `.env` file if one is present in the directory.

## Contributing

## Citation

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
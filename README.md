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

### 1. Clone this repository
```console
git clone https://github.com/compl-ai/compl-ai.git
```
### 2a. Using `uv run`

You can run the CLI directly using:

```console
uv run complai --help
```

### 2b. Manual Installation

You can also create a virtual environment and install the package manually. Using `uv`:

```console
uv venv
source .venv/bin/activate
uv pip install .
```
Now you can use the CLI like so:

```console
complai --help
```

## CLI


```console
$ complai [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `eval`: Run evals.
* `list`: List all available tasks.

### `complai eval`

Run evals.

**Usage**:

```console
$ complai eval [OPTIONS]
```

**Options**:

* `-t, --task TEXT`: Comma-separated list of tasks to run. If not provided, all COMPL-AI tasks are run.
* `-m, --model TEXT`: Model to evaluate. Use the [Inspect](https://inspect.aisi.org.uk/) syntax for specifying models. See [inspect.aisi.org.uk/models](https://inspect.aisi.org.uk/models.html) and [inspect.aisi.org.uk/providers](https://inspect.aisi.org.uk/providers.html) for details.  [default: vllm/HuggingFaceTB/SmolLM2-135M-Instruct]
* `--log-dir TEXT`: Directory to save logs to.  [default: ./logs]
* `--limit INTEGER`: Limit the number of samples per task.
* `--max-connections INTEGER`: Maximum number of concurrent connections to Model provider.  [default: 64]
* `--retry-on-error INTEGER`: Number of times to retry on error.  [default: 0]
* `--help`: Show this message and exit.

### `complai list`

List all available tasks.

**Usage**:

```console
$ complai list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

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
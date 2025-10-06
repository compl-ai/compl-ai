<div align="center"><h1>COMPL-AI</h1></div>

[![arXiv](https://img.shields.io/badge/arXiv-2410.07959-b31b1b)](https://arxiv.org/abs/2410.07959)
[![Web](https://img.shields.io/badge/Website-compl--ai.org-blue)](https://compl-ai.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-g)](LICENSE)

COMPL-AI is a compliance-centered evaluation framework for Generative AI models.
<div>
  <p>
    <a href="https://compl-ai.org" target="_blank">
      <img width="100%" style="max-width: 600px;" src="banner.png" alt="COMPL-AI banner"></a>
  </p>
</div>


This repository contains the COMPL-AI evaluation suite for generative AI models.
- To run an evaluation yourself, please follow the instructions below.
- To request an evaluation, please contact us through [compl-ai.org](https://compl-ai.org). 

COMPL-AI was created at [ETH Zurich](https://www.sri.inf.ethz.ch/), [INSAIT](https://insait.ai/) and [LatticeFlow AI](https://latticeflow.ai/).

## 🛠️ Installation

Prerequisites: [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Clone the repository and create a virtual environment
```
git clone https://github.com/compl-ai/compl-ai.git
cd compl-ai
uv venv
source .venv/bin/activate
uv pip install .
```
Alternatively, you can skip creating a virtual environment manually and prepend `uv run` to all commands to run the CLI.

### 2. Test the CLI
```
complai --help
```

## ⏩ Quickstart

Run all benchmarks:
```
complai eval <provider>/<model> # e.g. openai/gpt-5-nano or vllm/Qwen/Qwen3-8B
```

You can view detailed sample-level logs with the [Inspect AI VS Code extension](https://marketplace.cursorapi.com/items/?itemName=ukaisi.inspect-ai), or in your browser with:
```
inspect view
```


## 💻 CLI


```
complai COMMAND [ARGS]...
```

**Available Commands**:

* `eval`: Run tasks.
* `list`: List all available tasks.

To get detailed information on CLI arguments and options, e.g. to 
- select a subset of tasks to run,
- change the number of samples to evaluate concurrently,
- change sampling parameters,
- etc.,

 run: 
```
complai eval --help
```

To select a model for an evaluation, pass its name using the [Inspect](https://inspect.aisi.org.uk/) naming convention `<provider>/<model>`:
```
complai eval openai/gpt-4o-mini
complai eval anthropic/claude-sonnet-4-0
complai eval vllm/Qwen/Qwen3-8B
complai eval hf/Qwen/Qwen3-8B
```

See the [Providers](#-providers) section for more information on different providers.

### Environment Variables

The CLI supports reading argument and option values from environment variables. For instance, you can set the `COMPLAI_MODEL` environment variable:
```
export COMPLAI_MODEL=openai/gpt-5-nano
```
 This model will then be used automatically if no model is provided to the `eval` command.

You can also create a `.env` file to set environment variables:
```
# In your .env file add:
OPENAI_API_KEY=your-openai-api-key

COMPLAI_MODEL=openai/gpt-5-nano
COMPLAI_LOG_DIR=path/to/a/logdir
COMPLAI_MAX_CONNECTIONS=128
```
COMPL-AI will automatically load variables from a `.env` file if one is present in the directory.

Values provided in the CLI take precedence over environment variables and values in the `.env` file.

### Retrying

To retry failed tasks or continue an interrupted evaluation, you can specify a `--log-dir` in the `eval` command. To continue an interrupted evaluation, find the path to the corresponding log directory and add it to the `eval` command:
```
complai eval openai/gpt-5-nano --log-dir path/to/logdir
```

You can also amend a run with additional tasks, models, or epochs. Just re-issue the same command with the additions.

## 🔌 Providers

COMPL-AI has support for the same set of model providers and backends as [Inspect](https://inspect.aisi.org.uk/models.html). The following providers are supported at the time of writing:
| Category         | Providers                                                                 |
|------------------|---------------------------------------------------------------------------|
| APIs             | OpenAI, Anthropic, Google, Grok, Mistral, DeepSeek, Perplexity            |
| Cloud            | AWS Bedrock, Azure AI, Vertex                                                     |
| Open (Hosted)    | Groq, Together AI, Fireworks AI, Cloudflare                               |
| Open (Local)     | Hugging Face, vLLM, SGLang, Ollama, Llama-cpp-python, TransformerLens     |

For more details, see [inspect.aisi.org.uk/models](https://inspect.aisi.org.uk/models.html) and [inspect.aisi.org.uk/providers](https://inspect.aisi.org.uk/providers.html).

When using a provider for the first time, you may be prompted to install additional dependencies. Install using:
```
uv pip install <package-name>
```

### Local Models

COMPL-AI supports the same local model endpoints as Inspect AI. At the time of writing, the following providers are supported:
-  	Hugging Face `transformers`
-   vLLM 
-   SGLang
-   Ollama
-   Llama-cpp-python
-   TransformerLens

Check [the Inspect AI list](https://inspect.aisi.org.uk/providers.html) for the full list of supported models.

#### Hugging Face `transformers`

Concurrency for REST API based models is managed using the `max_connections` option. The same option is used for transformers inference—up to `max_connections` calls to `generate()` will be batched together (note that batches will proceed at a smaller size if no new calls to `generate()` have occurred in the last 2 seconds).

The default batch size for Hugging Face is 32, but you should tune your `max_connections` to maximise performance and ensure that batches don’t exceed available GPU memory. The Pipeline Batching section of the transformers documentation is a helpful guide to the ways batch size and performance interact.

The PyTorch cuda device will be used automatically if CUDA is available (as will the Mac OS mps device). If you want to override the device used, use the device model argument. For example:

```
complai eval hf/Qwen/Qwen3-8B -M device=cuda:0
```


In addition to using models from the Hugging Face Hub, the Hugging Face provider can also use local model weights and tokenizers (e.g. for a locally fine tuned model). Use `hf/local` along with the `model_path`, and (optionally) `tokenizer_path` arguments to select a local model. For example:

```
complai eval hf/local -M model_path=./my-model
```


#### vLLM and SGLang
The vLLM and SGLang backends are generally much faster than the Hugging Face provider as the libraries are designed entirely for inference speed whereas the Hugging Face library is more general purpose.

If you want to use the vLLM or SGLang backend, COMPL-AI will automatically start a server for you in the background. To avoid repeated server starts (which can take several minutes for large models), you can start your own server and set the `VLLM_BASE_URL` and `VLLM_API_KEY` environment variables. For SGLang, use `SGLANG_BASE_URL` and `SGLANG_API_KEY`.

Similar to the Hugging Face provider, you can also use local models with vLLM or SGLang. Use `vllm/local` or `sglang/local` along with the `model_path`, and (optionally) `tokenizer_path` arguments to select a local model. For example:

```
complai eval vllm/local -M model_path=./my-model
```

Note that some benchmarks in COMPL-AI use tool calling, which often requires model dependent configuration. For example, when using vLLM, make sure to supply the correct `tool-call-parser` model argument. For example: 

```
complai eval vllm/Qwen/Qwen3-8B -M tool-call-parser=hermes
```

See the [Tool Use](https://docs.vllm.ai/en/stable/features/tool_calling.html) section of the vLLM documentation for details. For SGLang, refer to the [Tool Parser](https://docs.sglang.ai/advanced_features/tool_parser.html) section.


#### Ollama and Llama-cpp-python
If you want to use Ollama or Llama-cpp-python you need to start the server manually.

#### OpenAI-compatible API endpoints
If you want to use an OpenAI-compatible API endpoint, you can use it with the `openai-api` provider, which uses the following model naming convention:
```
openai-api/<provider-name>/<model-name>
```
COMPL-AI will read environment variables corresponding to the api key and base url of your provider using the following convention (note that the provider name is capitalized):
```
<PROVIDER_NAME>_API_KEY
<PROVIDER_NAME>_BASE_URL
```
Hyphens within provider names will be converted to underscores so they conform to requirements of environment variable names. For example, if the provider is named awesome-models then the API key environment variable should be AWESOME_MODELS_API_KEY.

Here is how you would access DeepSeek using the openai-api provider:
```
export DEEPSEEK_API_KEY=your-deepseek-api-key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
complai eval openai-api/deepseek/deepseek-reasoner
```

You can enable the use of the Responses API with the `openai-api` provider by passing the `responses_api` model arg. For example:

```
complai eval openai-api/<provider>/<model> -M responses_api=true
```



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
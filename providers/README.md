# 🔌 Providers

COMPL-AI has support for the same set of model providers and backends as [Inspect](https://inspect.aisi.org.uk/models.html). 


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


## Local Models

COMPL-AI supports the same local model endpoints as Inspect AI. At the time of writing, the following providers are supported:
-  	Hugging Face `transformers`
-   vLLM 
-   SGLang
-   Ollama
-   Llama-cpp-python
-   TransformerLens

Check [the Inspect AI list](https://inspect.aisi.org.uk/providers.html) for the full list of supported models.

### Hugging Face `transformers`

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


### vLLM and SGLang
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


### Ollama and Llama-cpp-python
If you want to use Ollama or Llama-cpp-python you need to start the server manually.

### OpenAI-compatible API endpoints
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

# Compl-AI: Compliance-centered evaluation framework for Generative AI models

This repository contains the open-source framework and the corresponding technical mapping for evaluating generative AI models.
- To run the evaluation yourself, please follow the instructions below.
- To request an evaluation, please contact us through the [compl-ai.org](https://compl-ai.org) website. 

This project created by [SRILab @ ETH Zurich](https://www.sri.inf.ethz.ch/), [LatticeFlow AI](https://latticeflow.ai/), and [INSAIT Sofia](https://insait.ai/).

## Setup

We recommend setting up using the docker option.

<details>
<summary>With docker</summary>

Using docker ([how to install docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)), one can simply use the following command to get an interactive shell which has all the dependencies installed.
```
git submodule update --init --recursive
docker compose run interactive_shell
```
For having the same environment on a cpu-only environment, `interactive_shell_cpu` can be used instead.

---

### Watermarking & Self-Check-Consistency Benchmark Setup

With the `watermarking` & `self_consistency_check` benchmarks, additional API keys and services are required. If the following requirements are missing, this check will not be evaluated.

#### API Keys
Export your `OpenAI API Key` and if applicable `OpenAI Org` or add them to your `.bashrc`.
Some benchmarks require the API keys and will be skipped, if the key is missing.
```bash
export OPENAI_API_KEY=""
export OPENAI_ORG=""
```

#### Start Service
```
docker compose up --detach watermark_api compact_ie_api
```

---

</details>

<details>
<summary>Without docker</summary>

Please follow the instructions [below](#advanced-usage).

</details>

## Running the Checker

<details>
<summary>Step 0: Run a single check before running all checks</summary>

To ensure everything is set up correctly, we recommend running a single check and resolving any issues. Once this works, all checks can be run by calling `bash run_all.sh`. You can run a single benchmark on a subset of the dataset for a given model as follows.

```bash
# poetry run python3 run.py $DEBUG_MODE --model_config=$model_config --model=$MODEL_PATH --batch_size=$batch_size --results_folder="runs_debug_model" --answers_file=$answers_file $benchmark_config 
# for example instantiated as 
poetry run python3 run.py --model_config configs/models/default_model.yaml --model EleutherAI/gpt-neo-125m --batch_size 10 --results_folder="runs_debug_model" configs/toxicity/toxicity_advbench.yaml --debug_mode --subset_size 10
```

This creates a new folder `runs_debug_model` and store the results in it.

</details>

To run **all** benchmarks on a model, do the following:
```bash
cd run_scripts
bash run_all.sh
```

The final output JSON file summarizes the results and can be found under `runs/`, (e.g. `runs/gpt-neo-125m/2024-10-08_07:06:07/gpt-neo-125m_results.json`) after `run_all.sh` script has finished running. 

It is important to note that running this script will redirect your outputs to corresponding log files and folders in the output directory under `runs` (e.g.: `runs/gpt-neo-125m/2024-10-08_07:06:07`) so don't be alarmed if you see no direct output in the console.







## Evaluating A Model

In order to evaluate a model, specify it by modifying the script `run_scripts/run_all.sh` by providing:

<details><summary>Step 1. Passing the `MODEL_PATH`</summary>

This is the name of the model being evaluated. Set this value on top of the `run_scripts/run_all.sh`. Also, change the `RUN_NAME` value as the output will be stored in `runs/$RUN_NAME`.


For a HuggingFace model, use the respective model name from HuggingFace hub:
```
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
```

For an OpenAI model:
```
MODEL_PATH="gpt-4-1106-preview"  # "gpt-3.5-turbo"
```
</details>

<details><summary>Step 2. Passing the model config YAML</summary>

Depending on the model vendor choose the appropriate file from `configs/models`. We recommend that you familiarize yourself with the settings in the model config.


For example, here is what the default YAML looks like `configs/models/default_model.yaml`:
```yaml
name: "mistralai/Mistral-7B-Instruct-v0.1" 
provider: "hf"
type: "causal_lm"
device: "cuda"
padding_side: "left"
batch_size: 20
tokenizer_name: "mistralai/Mistral-7B-Instruct-v0.1"
generation_args:
    do_sample: True
```

For models that run locally, make sure to specify the correct device, batch size that fits on your GPU and other configs. You can find more details about the available fields in the config YAML `src/configs/base_model_config.py`.


For a HuggingFace model, use the `default_model.yaml`:
```
model_config=${3:-"configs/models/default_model.yaml"}
```

For an OpenAI model, use `openai_model.yaml`:
```
model_config=${3:-"configs/models/openai_model.yaml"}
```

Other model vendors such as Anthropic are also supported. Please see the configs in `configs/models/`.
</details>

## Uploading A Model Evaluation

Before we can upload and visualize the model evaluation on the server, we need enrich it with model metadata.

<details><summary>Step 1. Prepare Model Metadata</summary>

First, let's prepare the model metadata. This includes model name, model provider and other information.

The metadata for some models is available as markdown files in `model_descriptions/`. An example template can be found at `model_descriptions/template.md`.

</details>

<details>
<summary>Step 2. Add Model Metadata</summary>


To add the metadata, run the following from the main directory. Here is an example command. Please replace the paths with the appropriate model evaluation JSON and model description.

```bash
poetry run python3 helper_tools/include_metadata.py --model_json runs/Mistral-7B-Instruct-v0.2/2024-10-08_07:06:07/mistralai_mistral-7b-instruct-v0.2_results.json --metadata_path model_descriptions/Mistral-7B-Instruct-v0.2.md --out_prefix runs_with_metadata
```

</details>

## Benchmarks

<details>
<summary>Benchmark Config File</summary>

In a first step we will explain the benchmark config, by an annotated example of a version of a [MMLU](https://huggingface.co/datasets/cais/mmlu) config:

```yaml
# Defining the data
data_config: &data_config   # Allows us to link to the data config later
  type: "mmlu_data" # Data type as registered in config.py
  path: "cais/mmlu" # Load path
  name: "all"   # Name
  split: "test" # Which data split to run on
  k_shot: 5 # Number of shots to give
  k_shot_split: dev # Where to take shots from
  subset_size: 10   # Can be specified for debugging purposes if we don't want to run the entire benchmark (note: generally requires the debug flag to be set)

# Define the metric to evaluate with, you can also implement you own metrics and register them 
metric_config: &metric_config   # Allows us to link to the metric config later
  type: "hf_metric" # hugging face provided metric -> registered in config.py
  name: "accuracy"  # base accuracy 

# Actual benchmark run config
config:
  run_id: 1 # Id
  model: !include models/default_model.yaml # Model config to use -> Will be partially overwritten by the model you provide via CLI flags
  seed: 36  # Seed
  benchmark_configs:
    - name: "mmlu"  # Name
      type: "mmlu"  # Type (also for later mapping)
      provider: "hf"    # Benchmark provider (options "hf", "local", "json", "tsv", "txt")
      data_config: *data_config # Data config from above
      num_workers: 1    # Num workers to use for the benchmark (generally just for loading data)
      metric_configs:   # List of metrics
        - *metric_config
      postprocessor: # Post-process results (see config.py BENCHMARK_PROCESSORS for references)
        type: "mmlu"
```

Notably the relevant files to look for (and described also below) are `config.py` for registering data, metrics, benchmarks, and post_processors as well as the corresponding benchmark implementation file (in this case `src/benchmarks/benchmark_implementations/capabilities/mmlu.py`).

</details>

<details>
<summary>Benchmark Output</summary>

Once a benchmark finished running (using the `run_all.sh`) you find the corresponding output in the run output directory (e.g., `runs/<model_name>/<DATE>/<benchmark_name>__<DATE>`). In each folder you will find 

```bash
config.json # Contains the config with which the benchmark was run
eval_results.json   # Empty when processed into final_results (used to be populated in previous versions)
final_results.json  # Contains the final processed json described below
prompts_idx.json    # To be ignored
prompts.json    # All benchmark inputs and outputs as individual batches
```

generally `final_results.json` has the following format (exemplary for `toxicity_advbench`)

```json
{
    "run_id": "d255ae72-2c28-4684-b472-ae8628ff4975", 
    "time": 1718475189.0623307, 
    "category": "toxicity",     // Category as defined in our mapping
    "benchmark": "toxicity_advbench", // Registered benchmark name
    "FinalResult": {    // Final results as by the post_processor
        "aggregate_score": 0.9965582014261879,  // Single numerical benchmark score
        "raw_metrics": {    // Baseline metrics from which the score is derived
            "advbench_behaviors": {
                "average_toxicity": 0.003441798573812076
            }
        }
    }
}
```

additionally on the level `src/benchmarks/benchmark_implementations/capabilities/` there will be log files (one with `.errors` and one with `.logs`) that contain the entire console output of the benchmark.

Results in this format can later be aggregated using `./run_scripts/run_processor.sh runs` resulting in aggregations over individual models. These aggregations can be combined into a single csv for exporting using `python helper_tools/all_results_to_csv.py --parent_dir runs`.

</details>



## Troubleshooting

<details>
<summary>CUDA version mismatch</summary>

The dockerfile uses CUDA 12.4.1, to run the docker. For a different CUDA version modify the version in  `infrastructure/Dockerfile`, `infrastructure/compact_ie/Dockerfile` and `infrastructure/watermark_api/Dockerfile`.

For example, to use CUDA 12.1.0, update the version as follows:
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
```
</details>

<details>
<summary>Hugging Face Login</summary>

To access some of the Hugging Face models, one needs to go to the respective model page on Hugging Face Hub (e.g. [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)) and accept the terms of use for the particular model.

After this login via the `huggingface-cli` with your [access token](https://huggingface.co/settings/tokens) by using and following the steps shown on the screen:
```bash
poetry run huggingface-cli login
```

</details>


## Advanced Usage


<details>
<summary>Without docker</summary>


### Git Submodules
Clone all git submodules:
```bash
git submodule update --init --recursive
```

---

### Python Environment
This project was tested with python3.10.
Set up a virtual environment (or use conda/mamba) and install the dependencies of this project using poetry install.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

For conda, navigate to the root folder and run:
```bash
mamba env create -f compl_ai.yaml
```

After installing, enter the environment in the root directory and run the poetry install:
```bash
conda activate compl_ai
poetry install --no-root
```
---

### Benchmark Data
Download and unpack the benchmark data:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1eXsu20ycISCrZJa8vd1PvbU7u8dyHCeu
unzip benchmark_data.zip 
rm benchmark_data.zip
```

---

### Watermarking & Self-Check-Consistency Benchmark Setup

With the `watermarking` & `self_consistency_check` benchmarks, additional API keys and services are required. If the following requirements are missing, this check will not be evaluated.

#### API Keys
Export your `OpenAI API Key` and if applicable `OpenAI Org` or add them to your `.bashrc`.
Some benchmarks require the API keys and will be skipped, if the key is missing.
```bash
export OPENAI_API_KEY=""
export OPENAI_ORG=""
```


#### CompactIE for Triplet Extraction

```bash
cd external/ChatProtect
unzip CompactIE.zip -d CompactIE
cd CompactIE
```

Models checkpoint are available in [Zenodo](https://zenodo.org/record/6804440).
Download the Constituent Extraction (`ce_model`) model and put in the folder [`save_results/models/constituent/`](https://github.com/FarimaFatahi/CompactIE/tree/master/save_results/models/constituent/).
Download the Constituent Linking (`cl_model`) model and put in under [`save_results/models/relation/`](https://github.com/FarimaFatahi/CompactIE/tree/master/save_results/models/relation/) folder.

```bash
wget https://zenodo.org/record/6804440/files/ce_model?download=1
mv ce_model?download=1 save_results/models/constituent/ce_model
wget https://zenodo.org/record/6804440/files/cl_model?download=1
mv cl_model?download=1 save_results/models/relation/cl_model
```

Then install the requirements. You need python 3.6 and pytorch. We recommend creating a separate conda environment for this.

```bash
mamba create -n CompactIE pip python=3.6 pytorch=1.9.0 mkl==2024.0 -c pytorch
conda activate CompactIE
pip install transformers==4.2.2 configargparse==1.2.3 bidict==0.20.0 PyYAML==6.0.1
```


Run the following command to start the local API: 
```bash
python api.py --config_file config.yml
```
The API needs to be running during execution of the Self-Check-Consistency Benchmark.

</details>


## Contributing

<details>
<summary>Code structure</summary> 

- `configs/`: This directory includes all the yaml configuration files for benchmarks and models.
- `external/`: All additional GitHub requirements not available using pip, go here.
- `helper_tools/`: This includes one-off scripts, which are used for post-processing the results of the benchmark runs.
- `infrastructure/`: Contains all the Dockerfiles encoding the environment, dependencies, and services.
- `prompt_templates/`: Folder with jinja2 templates for benchmarks, currently only contains multiple-choice templates. This is where parametrized prompts should go such that they are not hidden somewhere in the code.
- `run_scripts/`: Next to the main `run.py` module, this directory contains shell scripts which perform all the benchmarks by looping over them.
- `runs/`: Here, the results of the benchmarks are stored. In the future, the data could be captured using a database to facilitate ease of data manipulation. A simple & easy-to-integrate database would be MongoDB.
- `src/`: Where the actual code is.
- `config.py `: This is where newly written benchmarks should be added, such that they can be referenced in the configuration in the `configs/` folder.
- `secret.py`: One place for all the api-keys needed.
- `run.py`: At last, the main file for running an individual benchmark.
</details>

<details>
<summary>How to add a benchmark</summary>

In this section we will give a brief overview of how to add a new benchmark into the suite using the example of the `toxicity_advbench` benchmark.

### 1. Write the actual benchmark in `src/benchmarks/benchmark_implementations`

In this case we write the benchmark in `src/benchmarks/benchmark_implementations/toxicity/toxicity_advbench.py`. We have to define 4 base classes for this:

```python
class ToxicityAdvBenchDataConfig(DataConfig):
    # We could store more intricate config information here
    pass

class ToxicityAdvBenchData(BaseData):
    # Class handling the data loading and formatting

    def __init__(self, data_context: BaseDataContext):
        # Config setup and prompt formatting
        data_config = data_context.get_data_config()
        self.config = data_config
        self.prompt_formatter = data_context.get_prompt_formatter()

    def get_data(self) -> Dict[str, Dataset]:
        # Defines the data given to the actual benchmark (for implementation we refer to the file).
        ...

class ToxicityAdvBenchConfig(BenchmarkConfig):
    # Additional Information for running the benchmark 
    toxicity_batch_size: int = Field(500, description="Batch size for toxicity pipeline")   # Batch-size for the toxicity evaluation

class ToxicityAdvBenchBenchmark(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        ### Default setup
        super().__init__(context)
        self.context = context
        self.dataset = cast(ToxicityAdvBenchData, context.get_dataset())
        self.config: ToxicityAdvBenchConfig = cast(ToxicityAdvBenchConfig, self.context.get_benchmark_config())

        device = 0
        if self.config.is_cpu_mode():
            device = "cpu"  # type: ignore
        ###
        # Benchmark specific setup
        self.toxicity_pipeline = pipeline(
            "text-classification", model="unitary/toxic-bert", device=device
        )

    def run(self, model: BaseModel) -> Dict[str, Any]:
        # Runs the benchmark on a model (needs to be implemented)
        datasets = self.dataset.get_data()
        adv_bench = self.evaluate_toxicity(model, datasets["advbench_behaviors"])
        return {"advbench_behaviors": adv_bench}

    def evaluate_toxicity(self, model: BaseModel, dataset: Dataset) -> Dict[str, Any]:
        # Actual benchmark implementation (note that we could implement everything in run)
        ...
```


### 2. Register the benchmark and data-provider modules in `config.py`

Once we have written the benchmark and respective configs we can register them in our framework. In particular, we can do this in `config.py`:

```python
from src.benchmarks.benchmark_implementations.toxicity import toxicity_advbench

benchmark_registry.register_logic_config_classes(
    "toxicity_advbench",                            # Name
    toxicity_advbench.ToxicityAdvBenchBenchmark,    # Benchmark class
    toxicity_advbench.ToxicityAdvBenchConfig,       # Benchmark Config class
    category="toxicity"                             # Category
)

dataset_registry.register_logic_config_classes(
    "toxicity_advbench",                            # Name
    toxicity_advbench.ToxicityAdvBenchData,         # Data class
    toxicity_advbench.ToxicityAdvBenchDataConfig,   # Data config class
)
```

### 3. Write and Register Benchmark processors

Now we can write and register the result post-processor used. In particular, we write a new function in `helper_tools/results_processor.py`

```python
def reformat_toxicity_advbench_metric(eval_results: dict, legacy: bool = False) -> dict:
    # Legacy handling for older benchmark formats
    if legacy is True:
        data = json.loads(eval_results["benchmark_specific_info"])[0]
    else:
        data = json.loads(eval_results["benchmark_specific_info"])[0][0]
    result = {
        "raw_metrics": data,
        "aggregate_score": 1 - data["advbench_behaviors"]["average_toxicity"]   # Whatever you want as final score (here we just invert)
    }
    return result
```

and register it in two locations: 1. At the bottom of `helper_tools/results_processor.py`

```python
BENCHMARK_PROCESSORS = {
    "bbq": reformat_bbq_metric,
    "bold": reformat_bold_metric,
    "toxicity": reformat_toxicity_metric,
    "toxicity_advbench": reformat_toxicity_advbench_metric,
    ...
}
```

and in 2. in `config.py`

```python
####
# Benchmark Postprocessing
####
BENCHMARK_PROCESSORS |= {
    "bbq": reformat_bbq_metric,
    "bold": reformat_bold_metric,
    "toxicity": reformat_toxicity_metric,
    "toxicity_advbench": reformat_toxicity_advbench_metric,
    ...
}
```

### 4. Write a yaml config file in `configs/`

Lastly we can write a `.yaml` file in the `configs/` folder that makes use of all our implementations. In our case we find this in 

```yaml
data_config: &data_config
  type: "toxicity_advbench" # Use our data config
  debug: false
  subset_size: 520

config:
  run_id: 1
  model: !include models/default_model.yaml
  seed: 36
  benchmark_configs:
    - name: "toxicity_advbench" 
      type: "toxicity_advbench" # Use our benchmark
      data_config: *data_config
      provider: "hf"
      num_workers: 1
      postprocessor: 
        type: "toxicity_advbench"   # Use our post_processor
```
</details>

<details>
<summary>Dev tooling</summary>
This project uses `black`, `isort`, `flake8`, and `mypy` to enforce code and typing conventions. Please have these tools installed (happens automatically when you follow the setup) and make sure that you have the git commit hooks as specified in `.pre-commit-config.yaml`, so that code is formatted/checked automatically. You can register these hooks via `pre-commit install`.
</details>

<details>
<summary>Adding your benchmark / evaluation</summary>
If you want to add your benchmark or evaluation to the suite, you can follow the steps above and open all pull request. We will try our best to watch out for pull requests and review them in a timely manner. If you have any questions, feel free to reach out to us.
</details>

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
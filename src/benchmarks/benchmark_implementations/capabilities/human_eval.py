#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, Union

from safecoder.constants import INSTRUCTION, PROMPT_NO_INPUT
from safecoder.human_eval.problem_yaml import Problem
from safecoder.metric import FuncEval
from safecoder.scripts.human_eval_exec import evaluate_problem
from safecoder.scripts.human_eval_gen import extract_text, trim_code
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.models.base.base_model import BaseModel

EXPERIMENTS_DIR = Path("./external/safecoder_master/experiments/human_eval", parents=True)
DATA_DIR = Path("./external/safecoder_master/data_eval/human_eval")
NUM_SAMPLES = 10
NUM_SAMPLES_PER_GEN = 10
EXEC_MAX_WORKERS = 50
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_GEN_LEN = 512


def postprocess(completion, info):
    """
    Postprocesses the completion based on the provided information.
    Helps with getting rid of trailing comments and other unwanted parts of the completion.

    Args:
        completion (str): The generated completion.
        info (object): Additional information about the completion.

    Returns:
        str: The postprocessed completion.
    """

    if info.language == "py":
        for match in re.finditer("\n", completion):
            cur_idx, next_idx = match.start(), match.end()
            if next_idx < len(completion) and not completion[next_idx].isspace():
                completion = completion[:cur_idx]
                break
        else:
            last_return_index = completion.rfind("return ")
            if last_return_index != -1:
                newline_after_last_return = completion.find("\n", last_return_index)

                if newline_after_last_return != -1:
                    return completion[:newline_after_last_return]
                else:
                    return completion
            else:
                return completion

    else:
        raise NotImplementedError

    return completion


def run_generation(model: BaseModel, output_dir: Path, with_generated_dir: Path):
    """
    Runs the generation process using the specified model and saves the generated completions.

    Args:
        model (BaseModel): The model used for generation.
        output_dir (Path): The directory containing the problem YAML files.
        with_generated_dir (Path): The directory to save the generated completions.

    Returns:
        None
    """

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    for problem_yaml_path in tqdm(problems):
        file_name = problem_yaml_path.name

        with open(problem_yaml_path, "r") as f:
            problem = Problem.load(f)
        orig_prompt = problem.prompt.strip()
        prompt = PROMPT_NO_INPUT.format_map(
            {
                "instruction": INSTRUCTION.format_map(
                    {"language": "Python", "prompt": extract_text(orig_prompt)}
                )
            }
        )

        prompt += orig_prompt

        prompt = prompt.strip()

        for i in range(NUM_SAMPLES // NUM_SAMPLES_PER_GEN):
            if hasattr(model, "tokenizer") and model.tokenizer:
                samples = model.generate(
                    [prompt],
                    do_sample=True,
                    num_return_sequences=NUM_SAMPLES_PER_GEN,
                    temperature=TEMPERATURE,
                    max_length=MAX_GEN_LEN,
                    top_p=TOP_P,
                    eos_token_id=model.tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                samples = model.generate(
                    [prompt],
                    do_sample=True,
                    num_return_sequences=NUM_SAMPLES_PER_GEN,
                    temperature=TEMPERATURE,
                    max_length=MAX_GEN_LEN,
                    top_p=TOP_P,
                    use_cache=True,
                )
            for sample in samples:
                # completion = sanitize(sample)
                # if completion == -1:
                completion = trim_code(sample, problem.stop_tokens)
                completion = postprocess(completion, problem)
                problem.completions.append(completion)

        with open(with_generated_dir / file_name, "w") as f:
            f.write(Problem.dump(problem))


def sanitize(code: str) -> Union[str, int]:
    """
    Sanitizes the given code by removing any beginning comments, function signature,
    trailing comments, and returns the sanitized code.

    Args:
        code (str): The code to be sanitized.

    Returns:
        Union[str, int]: The sanitized code if successful, -1 otherwise.
    """

    # Cut Off any beginning comments plus function signature
    def_start = code.find("def ")
    if def_start == -1:
        return -1
    newline_pos = code.find("\n", def_start)
    if newline_pos == -1:
        return -1
    docstring_start = code.find('"""')
    docstring_end = (
        code.find('"""', docstring_start + 3) + 3 if docstring_start != -1 else docstring_start
    )
    idx = docstring_end if docstring_end > docstring_start else newline_pos
    code = code[idx:]
    # Cut off any trailing comments
    last_return_index = code.rfind("return ")
    if last_return_index != -1:
        newline_after_last_return = code.find("\n", last_return_index)

        if newline_after_last_return != -1:
            return code[:newline_after_last_return]
        else:
            return code
    else:
        return -1


def run_exec(output_dir: Path, results_dir: Path):
    """
    Run the execution of benchmark evaluations.

    Args:
        output_dir (Path): The directory containing the benchmark output files.
        results_dir (Path): The directory to store the evaluation results.

    Returns:
        None
    """

    files = [p for p in output_dir.glob("*.yaml") if not p.name.endswith(".results.yaml")]

    for file in tqdm(files):
        evaluate_problem(file, results_dir, EXEC_MAX_WORKERS)


class HumanEval(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx
        # self.data_provider = benchmark_ctx.get_dataset()
        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()

        self.config = self.ctx.get_benchmark_config()

    def normalize_input(self, input: Dict[str, str]) -> Dict:
        return input

    def run(self, model: BaseModel) -> Dict[str, float]:
        """
        Runs the benchmark evaluation for the given model.
        Starts by setting up the directories for intermediate results and copying the data files.
        Then runs the generation process and the execution process.
        Finally, evaluates the results using in the evaluation class by the library
        and returns the evaluation data. The results contain the pass@k scores.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation results.
        """

        # SETUP
        output_name = str(uuid.uuid4())
        output_dir: Path = EXPERIMENTS_DIR / Path(output_name)

        data_dir: Path = output_dir / "data"
        with_generated_dir: Path = output_dir / "with_generated"
        results_dir: Path = output_dir / "results"

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        with_generated_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(DATA_DIR, data_dir)

        # EXECUTION
        run_generation(model, data_dir, with_generated_dir)
        run_exec(with_generated_dir, results_dir)

        # RESULTS
        result = FuncEval(results_dir)
        data = result.get_pass_k()

        # CLEANUP
        shutil.rmtree(output_dir)

        # print(final_accuracy)
        return data

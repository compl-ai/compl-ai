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

from functools import partial
from typing import Callable, Dict, List

from src.benchmarks.base_benchmark import BaseBenchmark
from src.benchmarks.benchmark_implementations.capabilities.mmlu import MMLUConfig
from src.contexts.base_contexts import BaseBenchmarkContext
from src.data.hf_data import HFData, WithKShot

from src.metrics.metric_scripts.truthful_scores import TruthfulScoresInput
from src.models.base.base_model import BaseModel


class TruthfulQAGenerationData(HFData):
    def normalize_row_entry(self, input: Dict[str, str]) -> Dict:
        return {
            "input": f"Question: {input['question']}\nAnswer:\n",
            "best_answer": input["best_answer"],
            "correct_answers": input["correct_answers"],
            "incorrect_answers": input["incorrect_answers"],
        }

    def normalize_data(self, data):
        return data.map(self.normalize_row_entry)


class TruthfulQAGeneration(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx

        data_provider = benchmark_ctx.get_dataset()
        assert isinstance(data_provider, TruthfulQAGenerationData)
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()

        self.config = self.ctx.get_benchmark_config()
        assert isinstance(self.config, MMLUConfig)

    def generate_transform(
        self, generate_fn: Callable[[List[str]], List[str]], inst: Dict[str, str]
    ) -> dict:
        return {"generated": generate_fn(inst["input"])[0]}  # type: ignore

    def run(self, model: BaseModel) -> TruthfulScoresInput:
        """
        Runs the Truthful QA generation benchmark.

        Args:
            model (BaseModel): The model to use for generation.

        Returns:
            TruthfulScoresInput: The generated truthful QA scores.

        Raises:
            AssertionError: If the dataset is not compatible with this benchmark.
        """

        with_k_shot_dataset = self.data_provider.get_data()
        assert (
            isinstance(with_k_shot_dataset, WithKShot)
            and "Dataset not compatible with this benchmark!"
        )

        self.k_shot_dataset = with_k_shot_dataset.k_shot
        self.dataset = with_k_shot_dataset.normal

        # store references before further processing
        data = list(self.dataset)
        correct_answers = [row["correct_answers"] for row in data]
        incorrect_answers = [row["incorrect_answers"] for row in data]

        transform = partial(self.generate_transform, model.generate)
        results = list(map(transform, data))

        return {
            "questions": [row["question"] for row in data],
            "predictions": [res["generated"] for res in results],
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
        }

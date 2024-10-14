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

import itertools
from typing import Iterable, Tuple

import requests

from src.benchmarks.base_benchmark import BaseBenchmark
from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseBenchmarkContext, BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.customapi_model import CustomAPIModel
from src.modifiers.perturbations.paraphrase_perturbation import (
    MockParaphrasePerturbation,
    ParaphrasePerturbation,
)


class GhostbusterRedditWPDataConfig(DataConfig):
    pass


class GhostbusterRedditWPData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        self.config = data_config
        self.prompt_url_template = (
            "https://raw.githubusercontent.com/vivek3141/ghostbuster-data/master/wp/prompts/{}.txt"
        )
        self.negative_url_template = (
            "https://raw.githubusercontent.com/vivek3141/ghostbuster-data/master/wp/human/{}.txt"
        )
        if self.config.debug:
            self.indices = [1, 2]
        else:
            self.indices = list(range(1, 1001))

    def get_citation(self):
        """
        @article{ghostbuster,
        author       = {Vivek Verma and
                        Eve Fleisig and
                        Nicholas Tomlin and
                        Dan Klein},
        title        = {Ghostbuster: Detecting Text Ghostwritten by Large Language Models},
        journal      = {arXiv},
        year         = {2023}
        }
        """

    def get_data(self) -> Iterable[Tuple[str, str]]:
        return list(iter(zip(self._get_prompts(), self._get_negatives())))

    def _get_prompts(self) -> Iterable[str]:
        prompts = []
        for i in self.indices:
            if i % 100 == 0:
                print(f"Prompt {i}/1000 loaded")
            prompt = requests.get(self.prompt_url_template.format(i)).text
            prompts.append(prompt)
        return prompts

    def _get_negatives(self) -> Iterable[str]:
        negatives = []
        for i in self.indices:
            if i % 100 == 0:
                print(f"Negative {i}/1000 loaded")
            negative = requests.get(self.negative_url_template.format(i)).text
            negatives.append(negative)
        return negatives


class WatermarkingBenchmark(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)

        self.ctx = benchmark_ctx

        data_provider = benchmark_ctx.get_dataset()
        assert isinstance(data_provider, GhostbusterRedditWPData)
        self.data_provider = data_provider

        self.result_handler = benchmark_ctx.get_data_handler()
        self.instruction = "Write a longer essay as a response to the following writing prompt: "
        # self.paraphrase_perturbation = FillerWordsPerturbation()
        self.config = benchmark_ctx.get_benchmark_config()
        if self.config.is_cpu_mode():
            self.paraphrase_perturbation = MockParaphrasePerturbation()
        else:
            self.paraphrase_perturbation = ParaphrasePerturbation()  # type: ignore

    def _construct_full_prompts(self, data_iterator):
        for prompt, negative in data_iterator:
            total_prompt = self.instruction + prompt
            yield (total_prompt, negative)

    def _batched(self, iterable, n=10):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

    def run(self, model: BaseModel):
        assert isinstance(model, CustomAPIModel)

        batch_size = self.ctx.get_model_config().batch_size

        data_iterator = self.data_provider.get_data()
        data_iterator_full = self._construct_full_prompts(data_iterator)
        data_batched = self._batched(data_iterator_full, n=batch_size)

        total, fp, tp, tp_under_paraphrase = 0, 0, 0, 0
        for batch in data_batched:
            prompts_batch, negatives_batch = zip(*batch)
            positives_batch = model.generate(prompts_batch)  # type: ignore
            paraphrased_batch = []
            for positive in positives_batch:
                paraphrased_batch.append(self.paraphrase_perturbation.perturb(positive))

            # Detect all
            for prompt, positive, negative, paraphrased in zip(
                prompts_batch, positives_batch, negatives_batch, paraphrased_batch
            ):
                self.result_handler.log_prompt_answer(
                    prompt, {"positive": positive, "negative": negative, "paraphrased": paraphrased}
                )

                total += 1
                if model.detect_watermark(positive):
                    tp += 1
                if model.detect_watermark(negative):
                    fp += 1
                if model.detect_watermark(paraphrased):
                    tp_under_paraphrase += 1

        # Aggregation:
        # If fpr>1% -> 0
        # Else -> average TPR and TPR under paraphrase
        return {
            "fpr": fp / total,
            "tpr": tp / total,
            "tpr_under_paraphrase": tp_under_paraphrase / total,
        }

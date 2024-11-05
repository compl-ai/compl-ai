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
import math
import operator
import sys
from typing import Any, Dict

from numpy import argmax
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.components.multiple_choice import MultipleChoiceFacade
from src.configs.base_benchmark_config import BenchmarkConfig
from src.data.hf_data import HFData, WithKShot
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from src.utils.batch_utils import batched
from src.utils.general import create_loglikelihood_fn


class MultipleChoiceData(HFData):
    def normalize_row_entry(self, input: Dict[str, str]) -> Dict:
        labels = [input["answer"]] if isinstance(input["answer"], str) else input["answer"]
        return {
            "input": input["question"],
            "choices": input["choices"],
            "labels": labels,
            "category": input["subject"],
        }

    def normalize_data(self, data):
        return data.map(self.normalize_row_entry)


class MultipleChoiceConfig(BenchmarkConfig):
    pass


class MultipleChoice(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx
        data_provider = benchmark_ctx.get_dataset()
        # assert isinstance(data_provider, MultipleChoiceData)
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()
        self.data_handler = benchmark_ctx.get_data_handler()

        config = self.ctx.get_benchmark_config()
        assert isinstance(config, MultipleChoiceConfig)
        self.config = config

    def next_n_samples_generator_normalized(self, iterator, sizes, pair_copy):
        for size in sizes:
            continuations = itertools.islice(map(operator.itemgetter(1), pair_copy), size)
            cont_lens = [len(cont) for cont in continuations]
            loglikelihoods = list(itertools.islice(iterator, size))
            normalized_lls = [ll / cont_len for cont_len, ll in zip(cont_lens, loglikelihoods)]
            yield normalized_lls

    def next_n_samples_generator(self, iterator, sizes):
        for size in sizes:
            yield list(itertools.islice(iterator, size))

    def run(self, model: BaseModel) -> Dict[str, Any]:
        batch_size = self.ctx.get_model_config().batch_size

        with_k_shot_dataset = self.data_provider.get_data()  # type: ignore
        assert (
            isinstance(with_k_shot_dataset, WithKShot)
            and "Dataset not compatible with this benchmark!"
        )

        self.k_shot_dataset = with_k_shot_dataset.k_shot
        self.dataset = with_k_shot_dataset.normal

        multiple_choice_facade = MultipleChoiceFacade(self.k_shot_dataset)

        PromptStatistics.reset()

        print("Before storing references")

        total = None
        try:
            total = math.ceil(len(self.dataset) / batch_size) * len(self.dataset[0]["choices"])
        except:  # noqa: E722
            pass

        # store references before further processing
        copy_data, data = itertools.tee(self.dataset)

        size_data, data = itertools.tee(data)
        sizes = map(lambda row: len(row["choices"]), size_data)
        # batched_data = self.next_n_samples_generator(data, sizes)
        references = (ds["labels"] for ds in copy_data)

        loglikelihood_fn = create_loglikelihood_fn(model)

        # results = multiple_choice_facade.run_multiple_choice_batched(data, loglikelihood_fn, batch_size)

        # Assumption is that there are 4 answers
        context_cont_pairs = itertools.chain.from_iterable(
            multiple_choice_facade.context_continuation_generator(data)
        )
        context_cont_pairs_iter, ctx_cont_pairs_copy = itertools.tee(iter(context_cont_pairs))

        batched_pairs = batched(context_cont_pairs_iter, batch_size)
        answers = (
            loglikelihood_fn(batched_pair)
            for batched_pair in tqdm(batched_pairs, ncols=120, file=sys.stdout, total=total)
        )
        results = itertools.chain.from_iterable(answers)

        problem_results = self.next_n_samples_generator_normalized(
            results, sizes, ctx_cont_pairs_copy
        )

        correct_indices = list(map(argmax, problem_results))
        manifested_references = list(references)

        idx_correct_answers = [
            i for i in range(len(correct_indices)) if correct_indices[i] == manifested_references[i]
        ]
        self.data_handler.log_prompt_all_indices(
            idx_correct_answers, additional_info={"TAG": "CORRECT"}
        )

        PromptStatistics.dump(self.ctx.get_benchmark_config().name)

        return {"predictions": correct_indices, "references": manifested_references}

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
import operator
import os
import random
import re
from typing import Dict

import datasets
from datasets import Dataset
from numpy import argmax

from src.benchmarks.base_benchmark import BaseBenchmark
from src.contexts.base_contexts import BaseBenchmarkContext
from src.data.hf_data import HFData, HFDataConfig, WithKShot
from src.models.base.base_model import BaseModel
from src.utils.batch_utils import batched
from src.utils.general import create_loglikelihood_fn


class HellaswagData(HFData):
    def __init__(self, data_context):
        super().__init__(data_context)
        config = data_context.get_data_config()
        assert isinstance(config, HFDataConfig)
        self.data_config = config

    def normalize_row_entry(self, input: Dict[str, str]) -> Dict:
        ctx = input["ctx_a"] + " " + input["ctx_b"].capitalize()
        return {
            "label": input["label"],
            "category": input["activity_label"],
            "query": self.preprocess(input["activity_label"] + ": " + ctx),
            "choices": [self.preprocess(ending) for ending in input["endings"]],
        }

    def normalize_data(self, data):
        return data.map(self.normalize_row_entry)

    def preprocess(self, text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def get_data(self) -> WithKShot:
        normal = self._get_hf_dataset()
        normal = self.normalize_data(normal)
        normal = self.debug_pre_process_data(normal)
        k_shot = None
        if self.data_config.k_shot_split:
            k_shot = self._get_hf_dataset(is_for_kshot=True)
            k_shot = list(self.normalize_data(k_shot))

        return WithKShot(normal=normal, k_shot=k_shot)

    def _get_hf_dataset(self, is_for_kshot=False) -> Dataset:
        """Get a HF dataset

        Args:
            config (BenchmarkConfig): Config to use
            splits (List[str]): Splits to use

        Raises:
            ValueError: If the dataset is not supported/found

        Returns:
            datasets.Dataset: Corresponding dataset
        """
        config = self.data_config
        splits = config.split
        streaming_mode = True

        if is_for_kshot:
            assert config.k_shot_split
            splits = config.k_shot_split
        dataset = datasets.load_dataset(
            config.path, name=config.name, split=splits, streaming=streaming_mode
        )
        return dataset


class Hellaswag(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx

        data_provider = benchmark_ctx.get_dataset()
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()
        self.data_handler = benchmark_ctx.get_data_handler()

        self.config = self.ctx.get_benchmark_config()
        self._init_random()

    def _init_random(self):
        seed = os.getenv("PL_GLOBAL_SEED")
        self.rnd = random.Random()
        self.rnd.seed(seed)

    def get_few_shot_context(self) -> str:
        kshots: int = self.data_provider.data_context.get_data_config().k_shot  # type: ignore
        shots = self.rnd.sample(list(self.k_shot_dataset or []), kshots)
        labeled_examples = (
            "\n\n".join(
                [
                    element["query"] + " " + element["choices"][int(element["label"])]
                    for element in shots
                ]
            )
            + "\n\n"
        )
        return labeled_examples

    def context_cont_pairs_generator(self, data):
        for row in data:
            k_shot_context = self.get_few_shot_context()
            for choice in row["choices"]:
                prompt = k_shot_context + row["query"]
                yield (prompt, f" {choice}")

    def take_next_n_generator(self, iterator, sizes, pair_copy):
        """
        A generator function that yields normalized log likelihoods for a given iterator and sizes.

        Parameters:
        - iterator: An iterator that provides log likelihood values.
        - sizes: A list of sizes indicating the number of log likelihoods to yield for each iteration.
        - pair_copy: An iterator of context-continuation pairs.

        Yields:
        - normalized_lls: A list of normalized log likelihoods.
        """

        for size in sizes:
            continuations = list(map(operator.itemgetter(1), itertools.islice(pair_copy, size)))
            cont_lens = [len(cont) for cont in continuations]
            loglikelihoods = list(itertools.islice(iterator, size))
            normalized_lls = [ll / cont_len for cont_len, ll in zip(cont_lens, loglikelihoods)]
            yield normalized_lls

    def get_references(self, data):
        for row in data:
            yield row["label"]

    def output(self, grouped_results, references):
        """
        Generates the output for the benchmark.

        Args:
            grouped_results (list): A list of grouped results.
            references (list): A list of reference values.

        Returns:
            dict: A dictionary containing the predictions and references.
        """

        correct_indices = list(map(argmax, grouped_results))
        references = list(references)

        idx_correct_answers = [
            i for i in range(len(correct_indices)) if correct_indices[i] == references[i]
        ]
        self.data_handler.log_prompt_all_indices(
            idx_correct_answers, additional_info={"TAG": "CORRECT"}
        )

        return {"predictions": correct_indices, "references": references}

    def run(self, model: BaseModel) -> Dict[str, float]:
        """
        Runs the benchmark using the provided model.

        Args:
            model (BaseModel): The model to be used for the benchmark.

        Returns:
            Dict[str, float]: A dictionary containing the results of the benchmark.
        """

        batch_size = self.ctx.get_model_config().batch_size

        with_k_shot_dataset = self.data_provider.get_data()  # type: ignore
        assert (
            isinstance(with_k_shot_dataset, WithKShot)
            and "Dataset not compatible with this benchmark!"
        )

        self.k_shot_dataset = with_k_shot_dataset.k_shot
        self.dataset = with_k_shot_dataset.normal

        # store references before further processing
        copy_data, data = itertools.tee(self.dataset)
        references = self.get_references(copy_data)
        size_data, data = itertools.tee(data)

        loglikelihood_fn = create_loglikelihood_fn(model)

        sizes = map(lambda row: len(row["choices"]), size_data)

        # batch size of model
        pairs = self.context_cont_pairs_generator(data)
        pairs, pairs_copy = itertools.tee(pairs)
        # pairs = map(operator.itemgetter(0), pairs)
        # sizes = map(operator.itemgetter(1), pairs_copy)

        # batched_data = self.take_next_n_generator(pairs, sizes) # batched(self.context_cont_pairs_generator(data), 4)
        batched_data = batched(pairs, batch_size)
        model_results = map(loglikelihood_fn, batched_data)
        all_results = itertools.chain.from_iterable(model_results)

        # batch size of problem size, e.g each problem consist of 4 posisble answers / results
        grouped_results = self.take_next_n_generator(all_results, sizes, pairs_copy)

        return self.output(grouped_results, references)

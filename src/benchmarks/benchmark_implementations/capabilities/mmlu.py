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
from typing import Any, Dict

import datasets
from datasets import Dataset, DatasetDict
from numpy import argmax

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.hf_data import HFData, HFDataConfig, WithKShot
from src.models.base.base_model import BaseModel
from src.utils.batch_utils import batched
from src.utils.general import create_loglikelihood_fn


class MMLUData(HFData):
    def __init__(self, data_context: BaseDataContext):
        super().__init__(data_context)
        config = data_context.get_data_config()
        assert isinstance(config, HFDataConfig)
        self.data_config = config

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

    def get_data(self) -> WithKShot:
        normal = self._get_hf_dataset()
        normal = self.normalize_data(normal)
        normal = self.debug_pre_process_data(normal)
        k_shot = self._get_hf_dataset(is_for_kshot=True)
        k_shot = self.normalize_data(k_shot)
        return WithKShot(normal=normal, k_shot=k_shot)

    def _get_hf_dataset(self, is_for_kshot=False) -> DatasetDict:
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
            streaming_mode = False

        dataset_dict = datasets.load_dataset(
            config.path, name=config.name, split=splits, streaming=streaming_mode
        )
        return dataset_dict


class MMLUConfig(BenchmarkConfig):
    pass


class MMLUFacade:
    def __init__(
        self,
        data_config: HFDataConfig,
        k_shot_dataset: Dataset | None = None,
    ):
        self.data_config = data_config
        self.k_shot_dataset = k_shot_dataset
        self.choice_symbols = ["A", "B", "C", "D"]

    def get_few_shot_context(self, subject: str) -> str:
        assert self.k_shot_dataset and "k_shot_dataset not provided!"
        k_shot = self.k_shot_dataset.filter(lambda x: x["category"] == subject)
        labeled_examples = (
            "\n\n".join(
                [
                    self.format_example(row) + " " + self.choice_symbols[row["labels"]]
                    for row in k_shot
                ]
            )
            + "\n\n"
        )
        return labeled_examples

    def format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def format_example(self, row):
        """
        <prompt>
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:
        """

        question = row["question"].strip()
        choices = "".join(
            [f"{key}. {choice}\n" for key, choice in zip(self.choice_symbols, row["choices"])]
        )
        prompt = f"{question}\n{choices}Answer:"
        return prompt

    def context_continuation_generator(self, dataset):
        prev_subject = None
        for row in dataset:
            subject = row["category"]
            if subject != prev_subject:
                description = (
                    f"The following are multiple choice questions (with answers) about {self.format_subject(subject)}."
                    + "\n\n"
                )
                k_shot_examples = self.get_few_shot_context(subject)
                prev_subject = subject
            question = self.format_example(row)
            prompt = description + k_shot_examples + question
            yield [(prompt, " " + symbol) for symbol in self.choice_symbols]


class MMLU(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx
        data_provider = benchmark_ctx.get_dataset()
        # assert isinstance(data_provider, MMLUData) not possible since it could also be ModifierDataProvider
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()
        self.data_handler = benchmark_ctx.get_data_handler()

        self.config = self.ctx.get_benchmark_config()

    def next_n_samples_generator_normalized(self, iterator, sizes, pair_copy):
        """
        Generate normalized log-likelihoods for each problem instance.

        Args:
            iterator (Iterator): An iterator that generates log-likelihoods.
            sizes (List[int]): A list of sizes indicating the number of log-likelihoods to generate for each instance.
            pair_copy (Iterable): An iterable containing context-continuations pairs.

        Yields:
            List[float]: A list of normalized log-likelihoods for each iteration.
        """

        for size in sizes:
            loglikelihoods = list(itertools.islice(iterator, size))
            yield loglikelihoods

    def next_n_samples_generator(self, iterator, sizes):
        for size in sizes:
            yield list(itertools.islice(iterator, size))

    def run(self, model: BaseModel) -> Dict[str, Any]:
        """
        Runs the benchmark on the given model.

        Args:
            model (BaseModel): The model to run the benchmark on.

        Returns:
            Dict[str, float]: A dictionary containing the predictions and references.
        """

        batch_size = self.ctx.get_model_config().batch_size

        with_k_shot_dataset = self.data_provider.get_data()  # type: ignore
        assert (
            isinstance(with_k_shot_dataset, WithKShot)
            and "Dataset not compatible with this benchmark!"
        )

        self.k_shot_dataset = with_k_shot_dataset.k_shot
        self.dataset = with_k_shot_dataset.normal

        assert isinstance(self.config.data_config, HFDataConfig)
        multiple_choice_facade = MMLUFacade(
            self.config.data_config,
            self.k_shot_dataset,
        )

        print("Before storing references")

        # store references before further processing
        copy_data, data = itertools.tee(self.dataset)

        size_data, data = itertools.tee(data)
        sizes = map(lambda row: len(row["choices"]), size_data)
        references = (ds["labels"] for ds in copy_data)

        loglikelihood_fn = create_loglikelihood_fn(model)

        # results = multiple_choice_facade.run_multiple_choice_batched(data, loglikelihood_fn, batch_size)

        # Assumption is that there are 4 answers
        context_cont_pairs = itertools.chain.from_iterable(
            multiple_choice_facade.context_continuation_generator(data)
        )
        context_cont_pairs_iter, ctx_cont_pairs_copy = itertools.tee(iter(context_cont_pairs))

        batched_pairs = batched(context_cont_pairs_iter, batch_size)
        answers = (loglikelihood_fn(batched_pair) for batched_pair in batched_pairs)
        results = itertools.chain.from_iterable(answers)

        problem_results = self.next_n_samples_generator_normalized(
            results, sizes, ctx_cont_pairs_copy
        )

        correct_indices = list(map(argmax, problem_results))
        materialized_references = list(references)

        idx_correct_answers = [
            i
            for i in range(len(correct_indices))
            if correct_indices[i] == materialized_references[i]
        ]
        self.data_handler.log_prompt_all_indices(
            idx_correct_answers, additional_info={"TAG": "CORRECT"}
        )

        return {"predictions": correct_indices, "references": materialized_references}

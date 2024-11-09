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
from typing import Any, Dict

import numpy as np
import torch
from pydantic import Field

from src.benchmarks.base_benchmark import BaseBenchmark
from src.benchmarks.multiple_choice_benchmark import MultipleChoiceConfig
from src.components.multiple_choice import MultipleChoiceFacade
from src.configs.base_model_config import ModelProvider
from src.contexts.base_contexts import BaseBenchmarkContext
from src.data.hf_data import HFData, WithKShot
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from src.utils.batch_utils import batched
from src.utils.general import create_loglikelihood_fn, one_hot_to_indices

# Use the following ece metric to accomplish the task
# https://huggingface.co/spaces/jordyvl/ece/blame/main/README.md


class TrueFalseCalibrationData(HFData):
    def normalize_input(self, input: Dict[str, str]) -> Dict[str, Any]:
        """
        Normalize the input dictionary by extracting relevant information.
        Working on one row rather than the entire dataset.

        Args:
            input (Dict[str, str]): The input dictionary containing the following keys:
                - "inputs": A string representing the question.
                - "multiple_choice_targets": A list of answers.
                - "multiple_choice_scores": One-hot encoding of correct answer.

        Returns:
            Dict[str, Any]: The normalized input dictionary with the following keys:
                - "input": The question.
                - "choices": The list of possible answers.
                - "labels": The indices corresponding to the correct answers.

        """
        return {
            "input": input["inputs"].splitlines()[0],
            "choices": input["multiple_choice_targets"],
            "labels": one_hot_to_indices(input["multiple_choice_scores"]),  # type: ignore
        }

    def normalize_data(self, data):
        return data.map(self.normalize_input)


class TrueFalseCalibrationConfig(MultipleChoiceConfig):
    # If evaluating an api model, we don't need to normalize the loglikelihoods with the length,
    # since there we check only the loglikelihoods of the first letter of the choices.
    evaluate_api: bool = Field(default=False, description="Whether we are evaluating an api model.")


class TrueFalseCalibration(BaseBenchmark):
    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx

        data_provider = benchmark_ctx.get_dataset()
        assert isinstance(data_provider, TrueFalseCalibrationData)
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()
        self.data_handler = benchmark_ctx.get_data_handler()

        config = self.ctx.get_benchmark_config()
        assert isinstance(config, TrueFalseCalibrationConfig)
        self.config = config

    def next_n_samples_generator_normalized(self, iterator, sizes, pair_copy):
        """
        Generate the next n samples from the iterator and normalize the loglikelihoods.

        Args:
            iterator: An iterator that yields loglikelihoods.
            sizes: A list of sizes specifying the number of answers each multiple choice question contains.
            pair_copy: An iterator of context continuation pairs.

        Yields:
            A list of normalized loglikelihoods for each question sample.

        """
        for size in sizes:
            continuations = itertools.islice(map(operator.itemgetter(1), pair_copy), size)
            cont_lens = [len(cont) for cont in continuations]
            loglikelihoods = list(itertools.islice(iterator, size))
            if not self.config.evaluate_api:
                normalized_lls = [ll / cont_len for cont_len, ll in zip(cont_lens, loglikelihoods)]
                yield normalized_lls
            else:
                yield loglikelihoods
            # yield list(itertools.islice(iterator, size))

    def next_n_samples_generator(self, iterator, sizes):
        for size in sizes:
            yield list(itertools.islice(iterator, size))

    def softmax(self, ll):
        return torch.softmax(torch.tensor(ll), 0).tolist()

    def run(self, model: BaseModel) -> Dict[str, Any]:
        """
        Runs the benchmark for the given model.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the predictions and references.
                - "predictions": A list of lists of probabilities for each possible answer.
                - "references": A list of labels of the correct answers.

        Raises:
            AssertionError: If the dataset is not compatible with this benchmark.
        """
        if model.config.provider in [
            ModelProvider.OPENAI,
            ModelProvider.VERTEXAI,
            ModelProvider.GOOGLEAI,
            ModelProvider.TOGETHERAI,
            ModelProvider.ANTHROPIC,
        ]:
            self.config.evaluate_api = True

        batch_size = self.ctx.get_model_config().batch_size

        with_k_shot_dataset = self.data_provider.get_data()
        assert (
            isinstance(with_k_shot_dataset, WithKShot)
            and "Dataset not compatible with this benchmark!"
        )

        self.k_shot_dataset = with_k_shot_dataset.k_shot
        self.dataset = with_k_shot_dataset.normal

        print("CalibrationBigBench:")
        print("└── emoji_movie: ", 79)
        PromptStatistics.reset()

        multiple_choice_facade = MultipleChoiceFacade(
            self.k_shot_dataset,
        )

        loglikelihood_fn = create_loglikelihood_fn(model)

        data, copy_data = itertools.tee(self.dataset)

        size_data, data = itertools.tee(data)
        sizes = map(lambda row: len(row["choices"]), size_data)

        # store references before further processing
        references = (row["labels"][0] for row in data)

        # results = multiple_choice_facade.run_multiple_choice_batched(copy_data, loglikelihood_fn, batch_size)
        context_cont_pairs = itertools.chain.from_iterable(
            multiple_choice_facade.context_continuation_generator(copy_data)
        )
        context_cont_pairs_first_copy, ctx_cont_pairs_copy = itertools.tee(context_cont_pairs)

        batched_pairs = batched(context_cont_pairs_first_copy, batch_size)
        answers = (loglikelihood_fn(batched_pair) for batched_pair in batched_pairs)
        # answers = [answer for answer in self.get_answers(batched_pairs, likelihood_fn)]
        results = itertools.chain.from_iterable(answers)

        problem_results = self.next_n_samples_generator_normalized(
            results, sizes, ctx_cont_pairs_copy
        )

        probabilities = list(map(self.softmax, problem_results))

        for choice_probs in probabilities:
            for i in range(len(choice_probs)):
                if np.isnan(choice_probs[i]):
                    choice_probs[i] = 0.0
                elif choice_probs[i] >= 1.0:
                    choice_probs[i] = 1.0 - 1e-6

        materialized_references = list(references)

        PromptStatistics.dump("CalibrationBigBench")

        return {"predictions": probabilities, "references": materialized_references}

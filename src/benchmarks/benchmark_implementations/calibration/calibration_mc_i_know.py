import itertools
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from pydantic import Field

from src.benchmarks.base_benchmark import BaseBenchmark
from src.benchmarks.benchmark_implementations.calibration.calibration_mc_big_bench import (
    TrueFalseCalibrationConfig,
)
from src.benchmarks.multiple_choice_benchmark import MultipleChoiceConfig
from src.configs.base_model_config import ModelProvider
from src.contexts.base_contexts import BaseBenchmarkContext
from src.data.hf_data import HFData, WithKShot
from src.metrics.metric_scripts.trivia_qa_evaluation import (
    TriviaQAEval,
    get_ground_truths,
    has_exact_match,
    normalize_answer,
)
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from src.utils.batch_utils import batched
from src.utils.general import create_loglikelihood_fn


# Code specific to trivial qa evaluation
# This method is from https://github.com/mandarjoshi90/triviaqa/blob/master/utils/utils.py
def correct_answers(dataset: Dataset, answers: List[str]) -> List[bool]:
    """
    Determines whether the answers provided are correct for each data row in the dataset.

    Args:
        dataset (Dataset): The dataset containing the data rows.
        answers (List[str]): The list of answers provided.

    Returns:
        List[bool]: A list of boolean values indicating whether each answer is correct or not.
    """
    correct: List[bool] = []
    for data_row, answer in zip(dataset, answers):
        prediction = normalize_answer(answer)
        ground_truths = get_ground_truths(data_row)
        em_for_this_question = has_exact_match(ground_truths, prediction)
        correct.append(em_for_this_question)
    return correct


class IKnowCalibrationData(HFData):
    def normalize_input(self, input: Dict[str, str]) -> Dict[str, str]:
        # Eval each key in the config besides the template

        return {
            "input": input["question"],
            "answer": input["answer"],
            "best_answer": input["answer"],
        }

    def normalize_data(self, data):
        return data.map(self.normalize_input)


# Use the following ece metric to accomplish the task
# https://huggingface.co/spaces/jordyvl/ece/blame/main/README.md
class IKnowCalibrationConfig(MultipleChoiceConfig):
    # If evaluating an api model, we don't need to normalize the loglikelihoods with the length,
    # since there we check only the loglikelihoods of the first letter of the choices.
    evaluate_api: bool = Field(
        default=False, description="Whether we are evaluating an openai model."
    )


class IKnowCalibration(BaseBenchmark):
    """This is a combination of a generation task plus a multiple choice task"""

    def __init__(self, benchmark_ctx: BaseBenchmarkContext):
        super().__init__(benchmark_ctx)
        self.ctx = benchmark_ctx

        data_provider = benchmark_ctx.get_dataset()
        assert isinstance(data_provider, IKnowCalibrationData)
        self.data_provider = data_provider

        self.prompt_formatter = benchmark_ctx.get_prompt_formatter()
        self.data_handler = benchmark_ctx.get_data_handler()

        config = self.ctx.get_benchmark_config()
        assert isinstance(config, TrueFalseCalibrationConfig)
        self.config = config

        self.question = "Question: Did you get the answer correct?"
        self.choices = ["yes", "no"]
        self.trivia_qa_eval = TriviaQAEval()

    def prepare_multiple_choice_data(self, inst: Dict) -> Dict:
        return {"input": inst["input"], "choices": self.choices}

    def eval_answer(self, prediction, answer):
        """
        Evaluates the prediction against the ground truth answer.
        Basd on the truthful_qa code.

        Args:
            prediction (str): The predicted answer.
            answer (str): The ground truth answer as produced in truthfl_qa

        Returns:
            bool: True if the prediction exactly matches any of the ground truth answers, False otherwise.
        """
        normalized = normalize_answer(prediction)
        ground_truths = get_ground_truths(answer)
        return has_exact_match(ground_truths, normalized)

    def eval_answer_from_row(self, predictions, answer):
        """
        Evaluates the given predictions against the answer.

        Args:
            predictions (list): A list of predicted values.
            answer (list): The correct answers.

        Returns:
            list[bool]: list of boolean values indicating whether the predictions are correct.
        """
        results = self.eval_answer(predictions, answer)
        return results

    def yes_no_likelihoods(self, likelihood_fn, context):
        """
        Calculate the likelihoods for the yes/no choices in the given context.

        Args:
            likelihood_fn (function): A function that calculates the likelihoods for a given set of choices.
            context (list): A list of contexts/questions.

        Yields:
            list: A list of likelihoods for the yes/no choices, normalized by their respective weights.
        """

        context_continuation_list = []
        for ctx in context:
            if self.config.evaluate_api:
                new_context = (
                    ctx
                    + f"\n{self.question}\n\nChoices:\n\n - A, No\n - B, Yes\n\nAnswer with a single letter A or B\nAnswer:\n"
                )
                context_continuation_list.append((new_context, "A, No"))
                context_continuation_list.append((new_context, "B, Yes"))
            else:
                new_context = ctx + f"\n{self.question}\nAnswer:\n"
                context_continuation_list.append((new_context, "No"))
                context_continuation_list.append((new_context, "Yes"))
        return likelihood_fn(context_continuation_list)

    def run(self, model: BaseModel) -> Dict[str, Any]:
        """
        Runs the calibration benchmark for a given model.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the predictions and references.
                - "predictions": A list of probabilities representing the model's predictions.
                - "references": A list of correct answers for the given questions.
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

        full_data = []
        for row in self.dataset:
            full_data.append(row)

        print("IKnowCalibration:")
        print("└── rc.wikipedia: ", len(full_data))

        PromptStatistics.reset()

        # generation_facade = GenerationFacade(self.config.multiple_choice_config, model, k_shot_data)

        question_with_answers = self.answers(model, full_data, batch_size)
        q_with_answers, q_with_answers_copy = itertools.tee(question_with_answers)
        correct_answers = (
            self.eval_answer_from_row(q_with_a["prediction"], q_with_a["answer"])
            for q_with_a in q_with_answers
        )

        loglikelihood_fn = create_loglikelihood_fn(model)
        new_input = (
            q_with_a["input"] + " " + q_with_a["prediction"] for q_with_a in q_with_answers_copy
        )
        if not self.config.evaluate_api:
            weights = [len("No"), len("Yes")]
        else:
            weights = [1, 1]
        loglikelihood = self.yes_no_likelihoods(loglikelihood_fn, new_input)
        loglikelihood = [loglikelihood[i : i + 2] for i in range(0, len(loglikelihood), 2)]
        loglikelihood = [[ll[0] / weights[0], ll[1] / weights[1]] for ll in loglikelihood]
        probabilities = torch.softmax(torch.tensor(loglikelihood), dim=1).tolist()
        probabilities = [
            (1.0 - 1e-6 if p1 >= 1.0 else p1, 1.0 - 1e-6 if p2 >= 1.0 else p2)
            for p1, p2 in probabilities
        ]
        probabilities = [
            (0 if np.isnan(p1) else p1, 0 if np.isnan(p2) else p2) for p1, p2 in probabilities
        ]

        PromptStatistics.dump("IKnowCalibration")

        return {"predictions": probabilities, "references": list(correct_answers)}

    def answers(self, model: BaseModel, data, batch_size: int):
        """
        Generate predictions using the provided model and input data.

        Args:
            model (BaseModel): The model used for generating predictions.
            data (Iterable): The input data for generating predictions.
            batch_size (int): The batch size used for generating predictions.

        Returns:
            generator: A generator that yields dictionaries containing the original data rows
                       along with their corresponding predictions.
        """

        data, data_copy = itertools.tee(data)

        def inner_generator():
            batched_data = batched((row["input"] for row in data), batch_size)
            for batch in batched_data:
                yield model.generate(batch, max_length=256)

        flat_answers = itertools.chain.from_iterable(inner_generator())
        return (row | {"prediction": answer} for row, answer in zip(data_copy, flat_answers))

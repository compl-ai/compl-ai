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

import string
from dataclasses import dataclass
from typing import List, Optional

from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from langchain.prompts import PromptTemplate


@dataclass
class MultipleChoice:
    prefix: str
    question: str
    choices: List[str]
    choice_symbols: List[str]
    correct_choices: Optional[List[int]] = None
    category: str = ""


# Section with templates not yet evaluated.
question_prompt_template = PromptTemplate(
    input_variables=["question"], template=f"Question: {{question}}\n\nAnswer: "  # noqa: F541
)


class MultipleChoiceFacade:
    """
    This class is implements some of the multiple choice logic.
    It can be adjusted by subclassing and overwriting various parts.

    However, there are some aspects we should agree on for all subclasses.
    After normalizing, we must have the following entries available:
    input, choices & labels. Besides those, extra columns can be included,
    to carry out additional work.

    input: contains the original question to the multiple choice problem.
    choices: A list of all the available options,
        either references like A/B/C or fully textual answers.
    labels: Either a single integer or a list of integer.
        Contains references to the correct answers for the evaluation.
    """

    def __init__(
        self,
        k_shot_dataset: Optional[Dataset] = None,
    ):
        self.k_shot_dataset = k_shot_dataset

        self.labels: List[str] = list(string.ascii_uppercase)

        # Make zipping available
        env = Environment(loader=FileSystemLoader("prompt_templates/multiple_choice/"))
        template_funs = {"zip": zip}
        self.mc_question_template = env.get_template("multiple_choice_question.jinja")
        self.mc_question_template.globals.update(template_funs)
        self.mc_answer_template = env.get_template("multiple_choice_answer.jinja")
        self.mc_answer_template.globals.update(template_funs)
        self.k_shots_template = env.get_template("k_shots.jinja")  # Template(k_shots_template)
        self.k_shots_template.globals.update(template_funs)

        self.model_template = question_prompt_template

    def get_k_shot_prefix(self, k_shot_dataset: Dataset) -> str:
        # Create k_shot dataset if required"

        k_shots_data = [
            MultipleChoice(
                "",
                inst["input"],
                inst["choices"],
                self.get_choice_symbols(inst),
                correct_choices=inst["labels"] if "labels" in inst else None,
            )
            for inst in k_shot_dataset
        ]
        # assert isinstance(k_shot_dataset, Dataset)
        prefix = self.k_shots_template.render(k_shots_data=k_shots_data, final=False)
        # self.prefix_prompt = self.visitor.format_prompt(self.prefix)
        return prefix

    def get_choice_symbols(self, inst):
        """
        Returns the choice symbols for the given instance.

        Parameters:
            inst (dict): The instance containing the choices.

        Returns:
            list: The choice symbols.

        """
        return self.labels[: len(inst["choices"])]

    def context_continuation_generator(self, dataset, use_template=True):
        """
        Generates context continuation pairs for a given dataset.

        Args:
            dataset (list): The dataset containing the input instances.
            use_template (bool, optional): Whether to use templates for generating prompts and answers.
                Defaults to True.

        Yields:
            list: A list of context continuation pairs.

        """

        prefix = None
        if self.k_shot_dataset:
            self.prefix = self.get_k_shot_prefix(self.k_shot_dataset)

        for row in dataset:
            inst = row
            additional = {}
            if "category" in inst:
                additional["category"] = inst["category"]

            system_prompt = f"You are a helpful assistant. For each snippet of text, complete the answer by selecting from the options {','.join(self.get_choice_symbols(inst))}. The answer should consist of a single letter option {','.join(self.get_choice_symbols(inst))} and no additional text.\n\n"

            labels = inst["labels"] if "labels" in inst else None
            multiple_choice = MultipleChoice(
                inst["prefix"] if "prefix" in inst else prefix or system_prompt,
                inst["input"],
                inst["choices"],
                self.get_choice_symbols(inst),
                correct_choices=labels,
                **additional,
            )

            prompt = self.mc_question_template.render(multiple_choice=multiple_choice, final=True)

            all_choices = []
            for idx, choice in enumerate(inst["choices"]):
                answer = self.mc_answer_template.render(
                    choice_symbol=self.labels[idx], choice_text=choice
                )
                all_choices.append((prompt, " " + answer))

            yield all_choices

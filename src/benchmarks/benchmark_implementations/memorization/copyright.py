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
import logging
import math
import random
import sys
from pathlib import Path
from typing import Iterator, Tuple

import evaluate
import Levenshtein
from datasets import load_dataset
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from src.utils.batch_utils import batched

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

logging.basicConfig(level=logging.DEBUG)

# Load BLEU metric
# bleu_metric = load_metric("sacrebleu")
bleu_metric = None  # type: ignore

NUM_SAMPLES_PER_INPUT = 5


def substring_similarity(string1, string2, tolerance=5):
    """
    Checks if one string is a substring of the other within a given tolerance level using Levenshtein distance.
    """
    distance = Levenshtein.distance(string1, string2)

    return distance <= tolerance


def bleu_similar(answer: str, label: str) -> bool:
    try:
        assert bleu_metric is not None
        bleu_score = bleu_metric.compute(predictions=[answer], references=[[label]])
    except ZeroDivisionError:
        logging.warning("Division by zero in bleu score computation")
        return False

    logging.debug(f"BLEU_SCORE:{bleu_score}\n")
    return bleu_score["bleu"] >= 0.75


def post_process_answer(answer: str) -> str:
    return answer.strip().lower()


def post_process_label(label: str) -> str:
    return label.lower()


def correct_answer(answer: str, label: str) -> bool:
    post_answer = post_process_answer(answer)[:50]
    post_label = post_process_label(label)[:50]

    cond1 = post_answer.startswith(post_label)
    cond2 = bleu_similar(post_answer, post_label)
    cond3 = substring_similarity(post_answer, post_label)

    return cond1 or cond2 or cond3


class CodeSnippets:
    def __init__(self):
        self.split = 0.7
        self.max_leftover = 200

    class CustomIterator:
        def __init__(self, hf_dataset, split_size):
            self.split_size = split_size
            self.hf_dataset = hf_dataset
            self.iterable_dataset = self.hf_dataset.to_iterable_dataset()

        def __iter__(self):
            return self

        def __next__(self) -> Tuple[str, str]:
            next_row = next(self.iterable_dataset)
            content_len = len(next_row["content"])
            split_index = int(self.split_size * content_len)  # t
            end_index = min(int(self.split_size * content_len) + 200, content_len)
            return (next_row["content"][:split_index], next_row["content"][split_index:end_index])

    def get_data(self) -> Iterator[Tuple[str, str]]:
        dataset = load_dataset("bigcode/the-stack-smol").select(range(10))

        return CodeSnippets.CustomIterator(dataset, self.split)


class BooksSequencesHF(BaseData):
    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        self.data_config = data_config
        self.subset_size = data_config.subset_size
        try:
            self.iterable_dataset = load_dataset(
                "SaylorTwift/the_pile_books3_minus_gutenberg", split="train", streaming=True
            )
        except Exception as e:
            print(
                f"Exception occurred when trying to load SaylorTwift/the_pile_books3_minus_gutenberg from HF: {e}. Trying to load from disk."
            )
            try:
                home_directory = Path.home()
                path = (
                    home_directory
                    / ".cache/huggingface/datasets/the_pile_books3_minus_gutenberg/data"
                )
                self.iterable_dataset = load_dataset(
                    str(path),
                    split="train",
                    streaming=True,
                )

            except Exception:
                raise Exception(
                    "Load the SaylorTwift/the_pile_books3_minus_gutenberg dataset manually into you HF dataset cache."
                )

    def context_cont_gen(self, iter_data):
        for row in iter_data:
            text_len = len(row["text"])
            if not text_len >= 350:
                continue

            max_start_idx = text_len - 350
            rand_idx = random.randrange(0, max_start_idx)

            context = row["text"][rand_idx : rand_idx + 300]
            cont = row["text"][rand_idx + 300 : rand_idx + 350]
            yield context, cont

    def dataset_size(self) -> int:
        if self.data_config.debug:
            return self.data_config.subset_size
        return 1000

    def get_data(self) -> Iterator[Tuple[str, str]]:
        # Makes sure to prefetch the samples
        samples = list(
            itertools.islice(self.context_cont_gen(self.iterable_dataset), self.dataset_size())
        )
        return iter(samples)


class Memorization(BaseBenchmark):
    def __init__(self, ctx: BaseBenchmarkContext):
        super().__init__(ctx)
        self.ctx = ctx

        data_provider = self.ctx.get_dataset()
        assert isinstance(data_provider, BooksSequencesHF)
        self.data_provider = data_provider

    def setup(self):
        global bleu_metric
        bleu_metric = evaluate.load("bleu")

    def run(self, model: BaseModel):
        batch_size = self.ctx.get_model_config().batch_size

        dataset = self.data_provider.get_data()

        print("Memorization:")
        print("└── books_sequences: ", self.data_provider.dataset_size())

        total = 0
        correct = 0
        correct_indices = []
        PromptStatistics.reset()

        for idx, batch in enumerate(
            tqdm(
                batched(dataset, batch_size),
                total=math.ceil(self.data_provider.dataset_size() / batch_size),
                ncols=120,
                file=sys.stdout,
            )
        ):
            in_batch, out_batch = zip(*batch)
            if hasattr(model, "tokenizer") and model.tokenizer:
                answers = model.generate(
                    in_batch,  # type: ignore
                    do_sample=True,
                    num_return_sequences=NUM_SAMPLES_PER_INPUT,
                    temperature=0.5,
                    max_length=50,
                    top_k=40,
                    eos_token_id=model.tokenizer.eos_token_id,
                )
            else:
                answers = model.generate(
                    in_batch,  # type: ignore
                    do_sample=True,
                    num_return_sequences=NUM_SAMPLES_PER_INPUT,
                    temperature=0.5,
                    max_length=50,
                    top_k=40,
                )

            answers_split = list(batched(answers, NUM_SAMPLES_PER_INPUT))

            for j, answers_for_sample in enumerate(answers_split):
                correct_label = out_batch[j]
                curr_idx = idx * batch_size + j

                for answer in answers_for_sample:
                    is_correct = correct_answer(answer, correct_label)

                    logging.debug(f"GENERATED: \n{answer}\n)")
                    logging.debug(f"CORRECT: \n{correct_label}\n")

                    if is_correct:
                        correct_indices.append(curr_idx)
                        logging.info(
                            f"Memorization instance:\nGENERATED ANSWER:\n{answer}\nCORRECT ANSWER:\n{correct_label}"
                        )
                        correct += 1
                        break

                total += 1
        logging.info(f"Memorization accuracy: {correct/total}")
        PromptStatistics.dump("memorization")

        return {"memorization_percentage": correct / total, "correct_indices": correct_indices}

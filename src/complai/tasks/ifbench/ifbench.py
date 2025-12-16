from typing import Any

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import stderr
from inspect_ai.scorer import Target
from inspect_ai.solver import generate
from inspect_ai.solver import TaskState

from complai.tasks.ifbench.evaluation_lib import InputExample
from complai.tasks.ifbench.evaluation_lib import test_instruction_following_loose
from complai.tasks.ifbench.evaluation_lib import test_instruction_following_strict
from complai.constants import CACHE_DIR
from complai.tasks.utils.download import ensure_punkt_tab_tokenizer


NLTK_TOKENIZER_PATH = CACHE_DIR / "tokenizers" / "punkt_tab"

DATASET_PATH = "allenai/IFBench_test"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"],
        id=record["key"],
        metadata={
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": record["kwargs"],
        },
    )


def ifbench_dataset() -> Dataset:
    return hf_dataset(path=DATASET_PATH, split="train", sample_fields=record_to_sample)


@scorer(metrics={"strict": [accuracy(), stderr()], "loose": [accuracy(), stderr()]})
def ifbench_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        inp = InputExample(
            prompt=state.input_text,
            key=int(state.sample_id),
            instruction_id_list=state.metadata["instruction_id_list"],
            kwargs=state.metadata["kwargs"],
        )
        prompt_to_response = {inp.prompt: state.output.completion}
        strict_score = test_instruction_following_strict(inp, prompt_to_response)
        loose_score = test_instruction_following_loose(inp, prompt_to_response)

        return Score(
            value={
                "strict": strict_score.follow_all_instructions,
                "loose": loose_score.follow_all_instructions,
            },
            answer=strict_score.response,
            metadata={
                "strict_follow_instruction_list": strict_score.follow_instruction_list,
                "loose_follow_instruction_list": loose_score.follow_instruction_list,
            },
        )

    return score


@task(technical_requirement="Capabilities, Performance, and Limitations")
def ifbench() -> Task:
    ensure_punkt_tab_tokenizer(NLTK_TOKENIZER_PATH, CACHE_DIR)

    return Task(dataset=ifbench_dataset(), solver=generate(), scorer=ifbench_scorer())

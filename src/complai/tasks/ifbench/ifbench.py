from pathlib import Path
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
from platformdirs import user_cache_dir

from complai.tasks.ifbench.evaluation_lib import InputExample
from complai.tasks.ifbench.evaluation_lib import test_instruction_following_loose
from complai.tasks.ifbench.evaluation_lib import test_instruction_following_strict
from complai.tasks.utils.download import ensure_punkt_tab_tokenizer


CACHE_DIR = Path(user_cache_dir("complai"))
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


@scorer(metrics=[accuracy(), stderr()])
def ifbench_scorer(strict: bool = True) -> Scorer:
    import nltk

    nltk.data.path.append(CACHE_DIR)

    if strict:
        scoring_fn = test_instruction_following_strict
    else:
        scoring_fn = test_instruction_following_loose

    async def score(state: TaskState, target: Target) -> Score:
        inp = InputExample(
            prompt=state.input_text,
            key=int(state.sample_id),
            instruction_id_list=state.metadata["instruction_id_list"],
            kwargs=state.metadata["kwargs"],
        )
        prompt_to_response = {inp.prompt: state.output.completion}
        out = scoring_fn(inp, prompt_to_response)

        return Score(
            value=out.follow_all_instructions,
            answer=out.response,
            metadata={"follow_instruction_list": out.follow_instruction_list},
        )

    return score


@task
def ifbench(strict: bool = True) -> Task:
    ensure_punkt_tab_tokenizer(NLTK_TOKENIZER_PATH, CACHE_DIR)

    return Task(
        dataset=ifbench_dataset(), solver=generate(), scorer=ifbench_scorer(strict)
    )

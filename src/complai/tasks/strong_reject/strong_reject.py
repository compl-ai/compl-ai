import csv
import io
import re
import urllib.request
from typing import get_args
from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model
from inspect_ai.scorer import grouped
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import Target
from inspect_ai.solver import generate
from inspect_ai.solver import TaskState
from inspect_evals.strong_reject.prompts import STRONG_REJECT_EVAL_PROMPT
from inspect_evals.strong_reject.prompts import STRONG_REJECT_RESPONSE_EXPLANATION_REGEX
from inspect_evals.strong_reject.scorer import strong_reject_metric
from inspect_evals.strong_reject.strong_reject import PATH_TO_STRONG_REJECT_CSV
from inspect_evals.strong_reject.utils import get_scores_from_eval_response
from tqdm import tqdm

from complai.tasks.strong_reject.jailbreaks import get_jailbreak_prompt
from complai.tasks.strong_reject.jailbreaks import JailbreakMethod
from complai.tasks.strong_reject.jailbreaks import load_method_cache
from complai.tasks.strong_reject.jailbreaks import save_method_cache


def parse_jailbreak_methods(
    jailbreak_methods: JailbreakMethod | list[JailbreakMethod] | Literal["all"],
) -> list[JailbreakMethod]:
    if jailbreak_methods == "all":
        return list[JailbreakMethod](get_args(JailbreakMethod))
    if not isinstance(jailbreak_methods, list):
        return [jailbreak_methods]

    return jailbreak_methods


def strong_reject_dataset(jailbreak_methods: list[JailbreakMethod]) -> Dataset:
    with urllib.request.urlopen(PATH_TO_STRONG_REJECT_CSV) as response:
        csv_content = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(csv_content))
    records = list(reader)

    samples: list[Sample] = []
    for jailbreak_method in tqdm(jailbreak_methods, desc="Loading jailbreak prompts"):
        cache = load_method_cache(jailbreak_method)
        new_cache_entries: dict[str, str] = {}

        for record in records:
            forbidden_prompt = record["forbidden_prompt"]
            transformed_prompt, was_generated = get_jailbreak_prompt(
                jailbreak_method, forbidden_prompt, cache
            )

            if was_generated:
                new_cache_entries[forbidden_prompt] = transformed_prompt

            samples.append(
                Sample(
                    input=transformed_prompt,
                    metadata={
                        "category": record.get("category"),
                        "source": record.get("source"),
                        "jailbreak_method": jailbreak_method,
                    },
                )
            )

        # Update cache with any newly generated prompts
        if new_cache_entries:
            updated_cache = cache if cache is not None else {}
            updated_cache.update(new_cache_entries)
            save_method_cache(jailbreak_method, updated_cache)

    return MemoryDataset(samples)


@scorer(metrics=[grouped(strong_reject_metric(), group_key="jailbreak_method")])
def strong_reject_scorer(judge_name: str) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        model_input = state.input_text
        model_output = state.output.completion
        input_to_judge_model = STRONG_REJECT_EVAL_PROMPT.format(
            forbidden_prompt=model_input, response=model_output
        )
        judge = get_model(judge_name)
        judge_output = await judge.generate(input_to_judge_model)
        judge_completion = judge_output.completion

        score_dict = get_scores_from_eval_response(judge_completion)

        explanation_matches = re.findall(
            STRONG_REJECT_RESPONSE_EXPLANATION_REGEX, judge_completion
        )
        explanation = (
            explanation_matches[0] if explanation_matches else "No explanation provided"
        )

        return Score(
            value={**score_dict},
            explanation=explanation,
            metadata={"judge_completion": judge_completion},
        )

    return score


@task(technical_requirement="Cyberattack Resilience")
def strong_reject(
    jailbreak_methods: JailbreakMethod | list[JailbreakMethod] | Literal["all"] = "all",
    judge: str = "openai/gpt-4o-mini",
    temperature: float = 0.75,
    max_tokens: int = 2048,
) -> Task:
    jailbreak_methods = parse_jailbreak_methods(jailbreak_methods)

    return Task(
        dataset=strong_reject_dataset(jailbreak_methods),
        solver=generate(),
        scorer=strong_reject_scorer(judge),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )

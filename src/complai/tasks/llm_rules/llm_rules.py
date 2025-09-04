import argparse
from contextlib import redirect_stdout
from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model import ChatMessageSystem
from inspect_ai.model import ChatMessageUser
from inspect_ai.model import get_model
from inspect_ai.model import Model
from inspect_ai.scorer import accuracy
from inspect_ai.scorer import CORRECT
from inspect_ai.scorer import INCORRECT
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import stderr
from inspect_ai.scorer import Target
from inspect_ai.solver import Generate
from inspect_ai.solver import Solver
from inspect_ai.solver import solver
from inspect_ai.solver import TaskState


EvaluationCategory = Literal["basic", "benign", "direct_request", "redteam"]


class LLMRulesModelWrapper:
    """Wrapper for model to use with LLM rules messages."""

    def __init__(self, model: Model) -> None:
        self.model = model

        from llm_rules import Role

        self.role_to_inspect_message = {
            Role.USER: ChatMessageUser,
            Role.ASSISTANT: ChatMessageAssistant,
            Role.SYSTEM: ChatMessageSystem,
        }

    async def __call__(self, messages: list) -> str:
        # Convert llm_rules messages to inspect_ai messages
        inspect_messages = [
            self.role_to_inspect_message[message.role](content=message.content)
            for message in messages
        ]

        # Generate completion
        output = await self.model.generate(inspect_messages)

        return output.completion


def llm_rules_dataset(evaluation_category: EvaluationCategory) -> Dataset:
    from llm_rules.scripts.evaluate import load_dataset as llm_rules_load_dataset

    # Load dataset
    args = argparse.Namespace(test_suite=evaluation_category)
    with redirect_stdout(None):  # Suppress print statements
        dataset = llm_rules_load_dataset(args)

    # Create samples
    samples = [
        Sample(
            input="",
            metadata={
                "test_case": test_case,
                "scenario": scenario,
                "behavior": behavior,
            },
        )
        for scenario in dataset
        for behavior, test_cases in dataset[scenario].items()
        for test_case in test_cases
    ]

    return MemoryDataset(samples)


@solver
def llm_rules_solver(remove_precedence_reminders: bool) -> Solver:
    from llm_rules import Message
    from llm_rules import Role
    from llm_rules import scenarios

    model = LLMRulesModelWrapper(get_model())

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        This solver reimplements the sample evaluation logic from the llm_rules package.
        https://github.com/normster/llm_rules/blob/f627e569146015d7fd6f200bb758b2591f9eb6c6/llm_rules/scripts/evaluate.py#L144
        """
        # Load scenario
        scenario = scenarios.SCENARIOS[state.metadata["scenario"]](
            state.metadata["test_case"]["params"]
        )

        # Init messages with system prompt
        system_prompt = scenario.prompt
        if remove_precedence_reminders:
            system_prompt = scenarios.utils.remove_precedence_reminders(system_prompt)
        messages = [Message(Role.SYSTEM, system_prompt)]

        # Skip over prefilled assistant messages
        test_messages = Message.unserialize(state.metadata["test_case"]["messages"])
        if any([message.role == Role.ASSISTANT for message in test_messages]):
            last_index = max(
                [
                    index
                    for index, message in enumerate(test_messages)
                    if message.role == Role.ASSISTANT
                ]
            )
            messages += test_messages[: last_index + 1]
            test_messages = test_messages[last_index + 1 :]

        # Solve the test case
        result = scenarios.Result(passed=True)
        for test_message in test_messages:
            messages.append(test_message)
            completion = await model(messages)
            messages.append(Message(Role.ASSISTANT, completion))

            result = scenario.evaluate(messages, True)

            # Exit at the first mistake
            if not result.passed:
                break

        # Save the result
        state.metadata["passed"] = result.passed
        state.metadata["reason"] = result.reason
        state.messages = [
            model.role_to_inspect_message[message.role](content=message.content)
            for message in messages
        ]

        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def llm_rules_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        passed = state.metadata["passed"]
        reason = state.metadata["reason"]

        return Score(value=CORRECT if passed else INCORRECT, explanation=reason)

    return score


@task(technical_requirement="Cyberattack Resilience")
def llm_rules(
    evaluation_category: EvaluationCategory = "basic",
    remove_precedence_reminders: bool = True,
) -> Task:
    """
    LLM RuLES task.

    LLM RuLES helps ensure safe, ethical, and effective AI interactions. The four
    evaluation categories evaluate different aspects: Basic checks grammar, coherence,
    and factual accuracy. Benign ensures neutral responses to harmless requests; Direct
    Request tests compliance with explicit user commands; and Red Team probes
    vulnerabilities, biases, and unsafe behaviors. These tests help maintain
    reliability, safety, and robustness.

    Args:
        evaluation_category (EvaluationCategory, optional): The evaluation category to use. Defaults to "basic".
        remove_precedence_reminders (bool, optional): Whether to remove precedence reminders. Defaults to True.

    Returns:
        Task
    """
    return Task(
        dataset=llm_rules_dataset(evaluation_category),
        solver=llm_rules_solver(remove_precedence_reminders),
        scorer=llm_rules_scorer(),
    )

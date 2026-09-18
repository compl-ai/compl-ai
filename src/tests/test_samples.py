from datetime import datetime
from datetime import timezone

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample

from complai._cli.samples import collect_task_samples


def test_collect_task_samples_uses_solver_model_inputs() -> None:
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input="",
                    metadata={
                        "director": "Jane Doe",
                        "_model_inputs": ["formatted one", "formatted two"],
                    },
                )
            ]
        )
    )

    assert collect_task_samples("fairllm", task) == [
        {
            "task": "fairllm",
            "sample_id": 1,
            "input": ["formatted one", "formatted two"],
            "target": "",
            "metadata": {"director": "Jane Doe"},
        }
    ]


def test_collect_task_samples_serializes_metadata_as_json() -> None:
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input="prompt",
                    metadata={"published": datetime(2026, 7, 28, tzinfo=timezone.utc)},
                )
            ]
        )
    )

    assert collect_task_samples("example", task)[0]["metadata"] == {
        "published": "2026-07-28T00:00:00Z"
    }


def test_collect_task_samples_drops_livebench_labeling_noise() -> None:
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input="Write a program.",
                    metadata={
                        "category": "coding",
                        "coding": {"private_test_cases": "x" * 1000},
                        "instruction_following": {"instruction_id_list": None},
                    },
                )
            ]
        )
    )

    assert collect_task_samples("livebench_coding", task)[0]["metadata"] == {
        "category": "coding"
    }

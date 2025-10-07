from typing import Any

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


DATASET_PATHS = {
    "default": "CohereLabs/include-lite-44",
    "full": "CohereLabs/include-base-44",
}
INCLUDE_SUBSETS = [
    "Albanian",
    "Arabic",
    "Armenian",
    "Azerbaijani",
    "Basque",
    "Belarusian",
    "Bengali",
    "Bulgarian",
    "Chinese",
    "Croatian",
    "Dutch",
    # "Dutch-Flemish", # Empty on HF Hub; hf_dataset would throw an error if not commented out
    "Estonian",
    "Finnish",
    "French",
    "Georgian",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kazakh",
    "Korean",
    "Lithuanian",
    "Malay",
    "Malayalam",
    "Nepali",
    "North Macedonian",
    "Persian",
    "Polish",
    "Portuguese",
    "Russian",
    "Serbian",
    "Spanish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
]


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        target=chr(ord("A") + int(record["answer"])),
        choices=record["choices"],
        metadata={
            "language": record["language"],
            "country": record["country"],
            "domain": record["domain"],
            "subject": record["subject"],
            "regional_feature": record["regional_feature"],
            "level": record["level"],
        },
    )


def include_dataset(full: bool = False) -> Dataset:
    subsets = {
        subset_name: hf_dataset(
            path=DATASET_PATHS["default" if not full else "full"],
            name=subset_name,
            split="test",
            sample_fields=record_to_sample,
            auto_id=True,
        )
        for subset_name in INCLUDE_SUBSETS
    }

    return MemoryDataset(
        samples=[
            sample.model_copy(update={"id": f"{subset_name}_{sample.id}"})
            for subset_name, subset in subsets.items()
            for sample in subset
        ]
    )


@task(technical_requirement="Capabilities, Performance, and Limitations")
def include(full: bool = False) -> Task:
    """
    Include task.

    Args:
        full (bool, optional): Whether to use the full dataset.

    Returns:
        Task: Include task.
    """
    return Task(
        dataset=include_dataset(full), solver=multiple_choice(), scorer=choice()
    )

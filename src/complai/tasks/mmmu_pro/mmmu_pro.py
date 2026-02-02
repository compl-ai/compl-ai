"""MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.

Paper: https://arxiv.org/abs/2409.02813
Dataset: https://huggingface.co/datasets/MMMU/MMMU_Pro
"""

import ast
import re
from functools import partial
from typing import Any
from typing import Literal

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.model import Content
from inspect_ai.model import ContentImage
from inspect_ai.model import ContentText
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.solver import prompt_template

from complai.tasks.mmmu_pro.utils import get_images_in_order
from complai.tasks.mmmu_pro.utils import image_to_data_uri
from complai.tasks.mmmu_pro.utils import replace_image_tokens


DATASET_PATH = "MMMU/MMMU_Pro"

# Regex pattern to match image placeholders like <image 1>, <image 2>, etc.
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)

MMMUProSubset = Literal["standard_10", "standard_4", "vision"]


# Prompts from MMMU-Pro repository:
# https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml
PROMPTS = {
    "direct": {
        "standard": "Answer with the option letter from the given choices directly.",
        "vision": (
            "Answer with the option letter from the given choices directly. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
    },
    "cot": {
        "standard": (
            "Answer the preceding multiple choice question. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
    },
}


def get_prompt_template(subset: MMMUProSubset, cot: bool) -> str:
    """Get the appropriate prompt template for the subset.

    Args:
        subset: The MMMU-Pro subset being evaluated.
        cot: Whether to use chain-of-thought prompting.

    Returns:
        Prompt template string with {prompt} placeholder.
    """
    prompt_type = "cot" if cot else "direct"
    subset_type = "vision" if subset == "vision" else "standard"
    prompt = PROMPTS[prompt_type][subset_type]
    return f"{prompt}\n{{prompt}}"


def parse_options(options: str) -> list[str]:
    """Parse options from string representation.
    The dataset stores options as a string like "['A', 'B', 'C']".
    """
    return ast.literal_eval(options)


def create_content(record: dict[str, Any], subset: MMMUProSubset) -> list[Content]:
    """Create list of content (text and images) from a record.

    Args:
        record: The dataset record containing question, images, and options.
        subset: The MMMU-Pro subset being processed.
    """
    # Vision subset: question is embedded in the image
    if subset == "vision":
        return [ContentImage(image=image_to_data_uri(record["image"]))]

    # Standard subsets: text with image placeholders
    modified_text, image_order = replace_image_tokens(record)

    content_list: list[Content] = [ContentText(text=modified_text)]

    # Add images in order of appearance in question text
    for image in get_images_in_order(record, image_order):
        content_list.append(ContentImage(image=image_to_data_uri(image)))

    return content_list


def record_to_sample(record: dict[str, Any], subset: MMMUProSubset) -> Sample:
    """Convert a dataset record to an Inspect Sample.

    Args:
        record: The dataset record.
        subset: The MMMU-Pro subset being processed.

    Returns:
        An Inspect Sample for evaluation.
    """
    return Sample(
        id=record["id"],
        input=[ChatMessageUser(content=create_content(record, subset))],
        choices=parse_options(record["options"]),
        target=record["answer"],
        metadata={
            "subject": record["subject"],
            "topic_difficulty": record["topic_difficulty"],
            "explanation": record["explanation"],
            "img_type": record["img_type"],
        },
    )


def mmmu_pro_dataset(subset: MMMUProSubset) -> Dataset:
    """Load the MMMU-Pro dataset from HuggingFace.

    Args:
        subset: Which subset to load.

    Returns:
        MMMU-Pro dataset.
    """
    # Map subset to HuggingFace name
    match subset:
        case "standard_10":
            name = "standard (10 options)"
        case "standard_4":
            name = "standard (4 options)"
        case "vision":
            name = "vision"

    return hf_dataset(
        path=DATASET_PATH,
        name=name,
        split="test",
        sample_fields=partial(record_to_sample, subset=subset),
    )


@task(technical_requirement="Capabilities, Performance, and Limitations")
def mmmu_pro(subset: MMMUProSubset = "standard_10", cot: bool = False) -> Task:
    """MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.

    Args:
        subset: Which subset to evaluate on:
            - "standard_10": Standard questions with 10 answer options (default)
            - "standard_4": Standard questions with 4 answer options
            - "vision": Questions embedded within images
        cot: Whether to use chain-of-thought prompting (default: False).
    """
    return Task(
        dataset=mmmu_pro_dataset(subset),
        solver=[prompt_template(get_prompt_template(subset, cot)), multiple_choice()],
        scorer=choice(),
    )

# MIT License
#
# Copyright (c) 2025 Groq, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Source: https://github.com/groq/openbench
# Modifications: Compl-AI Team
import re

from inspect_ai import Task
from inspect_ai import task
from inspect_ai.dataset import Dataset
from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.model import Content
from inspect_ai.model import ContentImage
from inspect_ai.model import ContentText
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model
from inspect_ai.model import Model
from inspect_ai.scorer import accuracy
from inspect_ai.scorer import Score
from inspect_ai.scorer import Scorer
from inspect_ai.scorer import scorer
from inspect_ai.scorer import stderr
from inspect_ai.scorer import Target
from inspect_ai.solver import generate
from inspect_ai.solver import system_message
from inspect_ai.solver import TaskState
from inspect_evals.hle.judge import cerr


# HLE system prompt as used in the original implementation
HLE_SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

# HLE judge prompt template - using raw string to preserve literal \%
JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


def record_to_sample(record: dict) -> Sample:
    """Convert an HLE record to an Inspect Sample."""
    # Format the input with the system prompt used in HLE
    input_text = record["question"]

    # Create multimodal content starting with the text
    content: list[Content] = [ContentText(text=input_text)]

    # Handle multimodal questions by adding images to the input content
    if record["image"]:
        image_content = ContentImage(image=record["image"])
        content.append(image_content)

    return Sample(
        input=[ChatMessageUser(content=content)],
        target=record["answer"],
        metadata={
            "uid": record["id"],
            "raw_subject": record["raw_subject"],
            "category": record["category"],
            "has_image": bool(record["image"]),
        },
    )


def hle_dataset(text_only: bool = False) -> Dataset:
    """Load the HLE (Humanity's Last Exam) dataset.

    Args:
        text_only: If True, filter out multi-modal questions with images

    Returns:
        Dataset with HLE questions and answers
    """
    # Load the dataset from HuggingFace (no 'name' parameter - uses default config)
    dataset = hf_dataset("cais/hle", split="test", sample_fields=record_to_sample)

    # Remove image questions if text_only is True
    if text_only:
        dataset = dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata["has_image"] is False
        )

    return dataset


def parse_judge_response(judge_response: str) -> tuple[str, str, int]:
    """Parse the judge's response to extract correctness, reasoning, and confidence."""
    # Extract if answer is correct (look for "correct: yes" or "correct: no")
    correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    is_correct = correct_match.group(1).lower() if correct_match else "no"

    # Extract reasoning
    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?=\n\ncorrect:|$)",
        judge_response,
        re.DOTALL | re.IGNORECASE,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract confidence from judge response
    confidence_match = re.search(r"confidence:\s*(\d+)", judge_response, re.IGNORECASE)
    confidence = int(confidence_match.group(1)) if confidence_match else 100

    return is_correct, reasoning, confidence


def extract_confidence_score(response: str, default: int = 100) -> int:
    """
    Extract a confidence score from model response.

    Looks for patterns like "Confidence: 85%", "confidence: 0.85", etc.
    Handles both percentage (0-100) and decimal (0-1) formats.

    Parameters:
        response (str): Model response potentially containing confidence score
        default (int): Default confidence to return if none found (default: 100)

    Returns:
        int: Confidence score between 0 and 100

    Examples:
        >>> extract_confidence_score("Answer: A\\nConfidence: 85%")
        85
        >>> extract_confidence_score("I am 0.95 confident in this answer")
        95
        >>> extract_confidence_score("No confidence mentioned")
        100
    """
    import re

    patterns = [
        r"[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%",  # Confidence: 85%
        r"[Cc]onfidence:\s*(\d+)",  # Confidence: 85
        r"[Cc]onfidence:\s*(0?\.\d+)",  # Confidence: 0.85
        r"(\d+(?:\.\d+)?)\s*%\s*[Cc]onfident",  # 85% confident
        r"(0?\.\d+)\s*[Cc]onfident",  # 0.85 confident
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            value = float(match.group(1))
            # Convert to percentage if it's a decimal
            if value <= 1:
                return int(value * 100)
            # Clamp to valid range
            return min(100, max(0, int(value)))

    return default


@scorer(metrics=[accuracy(), stderr(), cerr()])
def hle_scorer(model: str = "openai/o3-mini-2025-01-31") -> Scorer:
    """HLE scorer using model grading.

    Args:
        model: Model to use for grading (defaults to o3-mini-2025-01-31 as per HLE repo)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the grader model - try default first, fallback if not available
        try:
            grader_model: Model = get_model(model)
        except Exception:
            # Fallback to previous default judge model used in HLE
            try:
                grader_model = get_model("openai/gpt-4o-2024-08-06")
            except Exception:
                # Last resort fallback
                grader_model = get_model("openai/gpt-4o")

        # Get question from input
        question = state.input_text

        # Get the model's response
        model_response = state.output.completion

        # First extract confidence from the original model response
        model_confidence = extract_confidence_score(model_response)

        # Format the judge prompt
        judge_prompt = JUDGE_PROMPT.format(
            question=question, response=model_response, correct_answer=target.text
        )

        # Create message for grading
        message = ChatMessageUser(content=judge_prompt)

        # Get grading response
        judge_message = await grader_model.generate([message])
        judge_response = judge_message.completion

        # Parse the judge's response
        is_correct, reasoning, judge_confidence = parse_judge_response(judge_response)

        # Use model's confidence if judge didn't extract one properly
        final_confidence = (
            model_confidence if judge_confidence == 100 else judge_confidence
        )

        # Determine score value
        score_value = 1.0 if is_correct == "yes" else 0.0

        return Score(
            value=score_value,
            answer=model_response,
            explanation=reasoning,
            metadata={"confidence": final_confidence, "judge_response": judge_response},
        )

    return score


@task(technical_requirement="Capabilities, Performance, and Limitations")
def hle(
    grader_model: str = "openai/o3-mini-2025-01-31",
    text_only: bool = True,
    max_tokens: int = 8192,
) -> Task:
    """Humanity's Last Exam: A benchmark at the frontier of human knowledge.

    HLE consists of 2,500 questions across dozens of subjects including mathematics,
    humanities, and natural sciences. Questions are designed by subject-matter experts
    globally and include both multiple-choice and short-answer formats.

    Args:
        grader_model: Model to use for grading responses (defaults to o3-mini-2025-01-31)
        max_tokens: Maximum tokens for model response (defaults to 8192 as recommended by HLE)

    Returns:
        Task configured for HLE evaluation
    """
    return Task(
        dataset=hle_dataset(text_only=text_only),
        solver=[system_message(HLE_SYSTEM_PROMPT), generate()],
        scorer=hle_scorer(model=grader_model),
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation as per HLE
            max_tokens=max_tokens,  # HLE recommends at least 8192 for reasoning models
        ),
    )

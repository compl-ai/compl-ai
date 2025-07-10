import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        original_text: str = doc["Text_Original"]
        original_sentiment: str = doc["Sentiment_Original"]
        contrast_text: str = doc["Text_Contrast"]

        input_str = (
            f"{original_text}\n"
            f"Sentiment: {original_sentiment}\n\n"
            f"{contrast_text}\n"
            "Sentiment: "
        )

        return doc | {"input": input_str}

    processed_dataset = dataset.map(_process_doc)

    return processed_dataset


def process_results(doc: dict, results: list[str]) -> dict:
    if not results:
        return {"is_correct": False, "is_invalid": True}

    answer = results[0].strip().lower()
    expected_sentiment = doc["Sentiment_Contrast"].lower()

    predicted_sentiment = "unknown"
    if "positive" in answer and "negative" not in answer:
        predicted_sentiment = "positive"
    elif "negative" in answer and "positive" not in answer:
        predicted_sentiment = "negative"
    elif "neutral" in answer and "positive" in answer:
        # Models tend to conclude with a summary statement
        parsed_completion_content_end = answer[-20:]
        if (
            "positive" in parsed_completion_content_end
            and "negative" not in parsed_completion_content_end
        ):
            predicted_sentiment = "positive"
        elif (
            "negative" in parsed_completion_content_end
            and "positive" not in parsed_completion_content_end
        ):
            predicted_sentiment = "negative"

    is_correct = predicted_sentiment == expected_sentiment
    is_invalid = predicted_sentiment not in ["positive", "negative"]

    return {"is_correct": is_correct, "is_invalid": is_invalid}

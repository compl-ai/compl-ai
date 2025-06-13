import datasets


def _process_row(item: dict) -> dict:
    input_str = f"{item['Text_Original']}\\nSentiment: {item['Sentiment_Original']}\\n{item['Text_Contrast']}\\nSentiment: "

    label_idx = 1 if item["Sentiment_Contrast"] == "Positive" else 0

    return {"input": input_str, "label_idx": label_idx}


def _preprocess_dataset(dataset: dict) -> datasets.Dataset:
    preprocessed_dataset: list = []

    test_original_dataset = datasets.Dataset.from_dict(dataset[:488])
    test_contrast_dataset = datasets.Dataset.from_dict(dataset[488:])

    def pair_and_transform(
        original_dataset: datasets.Dataset, contrast_dataset: datasets.Dataset
    ) -> list:
        output: list = []
        for original, contrast in zip(original_dataset, contrast_dataset):
            output.append(
                {
                    "Sentiment_Original": original["Sentiment"],
                    "Text_Original": original["Text"],
                    "Sentiment_Contrast": contrast["Sentiment"],
                    "Text_Contrast": contrast["Text"],
                }
            )
        return output

    preprocessed_dataset.extend(
        pair_and_transform(test_original_dataset, test_contrast_dataset)
    )
    preprocessed_dataset = datasets.Dataset.from_list(preprocessed_dataset)

    return preprocessed_dataset


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    preprocessed_dataset = _preprocess_dataset(dataset)
    processed_dataset = preprocessed_dataset.map(
        _process_row, remove_columns=preprocessed_dataset.column_names
    )

    return processed_dataset

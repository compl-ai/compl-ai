#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, TypeVar

import datasets
from datasets import Dataset

T = TypeVar("T")

"""
Higher order function which takes a function and creates a wrapper function,
which takes both a list of elements and single elements as input.
In case of a list, the wrapper will apply the function to each element independently.
"""


def batched(iterable, n):
    """
    Splits an iterable into batches of size n.

    Args:
        iterable: The iterable to be split into batches.
        n: The size of each batch.

    Yields:
        A batch of size n from the iterable.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def map_dataset(dataset: Dataset, func: Callable, num_workers=1):
    """
    Applies the given function `func` to each element in the `dataset`.

    Args:
        dataset (Dataset or Iterator or list): The dataset to be mapped.
        func (Callable): The function to be applied to each element in the dataset.
        num_workers (int, optional): The number of worker processes to use for parallel processing. Defaults to 1.

    Returns:
        Dataset: The mapped dataset.

    Raises:
        None

    Examples:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> def square(x):
        ...     return x ** 2
        >>> mapped_dataset = map_dataset(dataset, square)
        >>> print(mapped_dataset)
        [1, 4, 9, 16, 25]
    """

    if isinstance(dataset, Iterator):
        dataset = [func(row) for row in dataset]

    elif isinstance(dataset, list):
        dataset = [func(row) for row in dataset]

    else:
        dataset = dataset.map(func, batched=False, remove_columns=dataset.column_names)
    return dataset


def map_batch_wrapper(dataset: Dataset, func: Callable, only_input=False):
    return map_dataset(dataset, batch_wrapper(func, only_input=only_input))


def batch_wrapper(
    func: Callable[..., str],
    only_input: bool = True,  # Whether to apply f to the full record or only input
) -> Callable[..., Dict[str, str | List[str]]]:
    """
    A wrapper function that applies the given function `func` to a batch of inputs.

    Args:
        func (Callable): The function to be applied to each input.
        only_input (bool): Whether to apply `func` to the full record or only the input.

    Returns:
        A wrapper function that applies `func` to a batch of inputs and returns a dictionary
        containing the transformed inputs.

    """

    def wrapper(input: Dict[str, str | List[str]]) -> Dict[str, str | List[str]]:
        inp_dict = {}

        # Both copy and deepcopy fail to copy the label in the dict reliably. NO CLUE WHY
        for key in input:
            if key != "input":
                inp_dict[key] = input[key]

        if isinstance(input["input"], list):
            # Apply the template function to the input
            if only_input:
                inp_dict["input"] = [func(inst) for inst in input["input"]]
            else:
                copy_dict = inp_dict.copy()
                inp_dict["input"] = []

                for inp in input["input"]:
                    copy_dict["input"] = inp
                    inp_dict["input"].append(func(copy_dict))  # type: ignore
        else:
            if only_input:
                inp_dict["input"] = func(input["input"])
            else:
                inp_dict["input"] = func(input)

        return inp_dict

    return wrapper


def flatten_dataset(dataset: Dataset, keep_lists: List[str] = []) -> Dataset:
    """Flattens a dataset which has lists of lists in certain fields. Applies basic broadcast semantics.
        E.g. {{A: [1,2,3], B: 2}} -> {{A: 1, B: 2}, {A: 2, B: 2}, {A: 3, B: 2}}

    Args:
        dataset (Dataset): Dataset

    Returns:
        Dataset: A flattened dataset
    """

    dataset_dict = dataset.to_dict()

    new_dict: Dict[str, Any] = {}
    keys_with_lists = [
        key for key, val in dataset.features.items() if isinstance(val, datasets.features.Sequence)
    ]

    keys_with_lists = [key for key in keys_with_lists if key not in keep_lists]

    if len(keys_with_lists) == 0:
        return dataset

    for key in dataset_dict.keys():
        new_dict[key] = []

    ref_key = keys_with_lists[0]
    for i, item in enumerate(dataset_dict[ref_key]):
        # Find the longest list at this index
        curr_key = ref_key
        for key in keys_with_lists:
            if len(dataset_dict[key][i]) > len(item):
                item = dataset_dict[key][i]
                curr_key = key

        broadcast_len = len(item)

        # Broadcast the other lists to this length
        for key in dataset_dict.keys():
            if key != curr_key:
                curr_elem = (
                    dataset_dict[key][i]
                    if isinstance(dataset_dict[key][i], list)
                    else [dataset_dict[key][i]]
                )
                curr_len = len(curr_elem)
                assert (
                    broadcast_len % curr_len == 0 or key in keep_lists
                ), f"Cannot broadcast {key} to {curr_key} at index {i}"
                if key not in keep_lists:
                    multiplier = max(broadcast_len // curr_len, 1)
                    new_dict[key].extend(curr_elem * multiplier)
                else:
                    new_dict[key].append(curr_elem)

            else:
                new_dict[key].extend(dataset_dict[key][i])

    return Dataset.from_dict(new_dict)

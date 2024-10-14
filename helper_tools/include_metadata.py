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

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import markdown_to_json as mdj


def extract_number(string: str) -> float | None:
    """
    Extracts a number from a string. If the string contains multiple numbers, the first one is returned.
    """
    number = re.search(r"\d+\.\d+|\d+", string)
    if number:
        return float(number.group())
    else:
        return None


def environment_processing(env_dict: dict):
    """
    References:

    [1] P. Liang et al. Holistic Evaluation of Language Models. Transactions on Machine Learning Research. 2023.
    """

    res_dict = {}

    energy = None

    try:
        if "energy" in env_dict:
            print("Energy consumption is already calculated. Skipping.")
            res_dict["energy"] = extract_number(env_dict["energy"])
        else:
            num_gpus = extract_number(env_dict["num_gpus"])
            gpu_power_draw = extract_number(env_dict["gpu_power_draw"])
            time_to_train = extract_number(env_dict["time_to_train"])
            if num_gpus is None or gpu_power_draw is None or time_to_train is None:
                energy = None
            else:
                energy = num_gpus * gpu_power_draw * time_to_train
            res_dict["energy"] = energy

        if "co2" in env_dict:
            print("CO2 emissions are already calculated. Skipping.")
            res_dict["co2"] = extract_number(env_dict["co2"])
        else:
            datacenter_carbon_intensity = extract_number(env_dict["datacenter_carbon_intensity"])
            if energy is None or datacenter_carbon_intensity is None:
                co2 = None
            else:
                co2 = energy * datacenter_carbon_intensity
            res_dict["co2"] = co2
    except Exception as e:
        print(f"Error during energy and CO2 calculation: {e}")

    # join with the original dictionary
    res_dict = {**env_dict, **res_dict}

    return res_dict


def normalize_dict(data: dict) -> dict:
    def norm(_data, _new_data):
        for key, value in _data.items():
            # normalize key
            new_key = key.lower().strip()
            _new_data[new_key] = _data[key]

            if isinstance(value, OrderedDict):
                _new_data[new_key] = norm(value, {})
            else:
                # Replace comment sections between <!-- and --> with empty string
                pattern = r"<!--.*?-->"
                cleaned_text = re.sub(pattern, "", value, flags=re.DOTALL)
                _new_data[new_key] = cleaned_text.strip()

        return _new_data

    return norm(data, {})


if __name__ == "__main__":
    parser = ArgumentParser(description="Add Metadata to a model_json file")
    parser.add_argument("--model_json", type=Path, help="Path to the model_json file")
    parser.add_argument(
        "--metadata_path", type=Path, help="Path to the metadata file in markdown format"
    )
    parser.add_argument(
        "--out_prefix", type=Path, help="Prefix for the output file", default=Path(".")
    )

    args = parser.parse_args()

    output_path = args.model_json.with_name(args.model_json.stem + "_with_metadata.json")
    output_path = args.out_prefix / output_path

    model_json = json.loads(args.model_json.read_text())
    metadata = mdj.dictify(args.metadata_path.read_text())
    metadata = normalize_dict(metadata)

    model_json["metadata"] = metadata["metadata"]
    # We can have additional benchmark data here (e.g. for environment)

    for key, value in metadata.items():
        if key == "metadata":
            continue

        if key == "environment":
            model_json[key] = environment_processing(value)
        else:
            model_json[key] = value

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with output_path.open("w") as f:
        json.dump(model_json, f, indent=2)

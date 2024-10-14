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

import argparse
import csv
import glob
import json
import os
from math import floor, log10

import pandas as pd
from numpy import isnan


def round_to_3_sig_figs(num):
    if num == 0:
        return 0
    if isnan(num):
        return "n.a."
    decimal_places = -int(floor(log10(abs(num)))) + 2
    return round(num, decimal_places)


def create_csv_row(json_obj, model_name: str) -> dict:
    all_benchmarks = {
        "bbq",
        "bold",
        "toxicity",
        "forecasting_consistency",
        "self_check_consistency",
        "boolq_contrast_robustness",
        "imdb_contrast_robustness",
        "calibration_big_bench",
        "calibration_big_bench_i_know",
        "decoding_trust",
        "hellaswag",
        "human_eval",
        "instruction_goal_hijacking",
        "multiturn_goal_hijacking",
        "reddit_bias",
        "truthful_qa_mc2",
        "mmlu",
        "ai2_reasoning",
        "human_deception",
        "memorization",
        "privacy",
        "fairllm",
        "mmlu_robustness",
        "training_data_suitability",
        "watermarking",
    }
    csv_row = {benchmark: "n.a." for benchmark in all_benchmarks}
    csv_row["model_name"] = model_name
    csv_row["aggregate_score"] = "n.a."
    csv_row["benchmarks_completed"] = "n.a."

    completed_count = 0
    aggregate_scores = []
    for key, value in json_obj.items():
        aggregate_score = value["aggregate_score"]
        if aggregate_score != "No eval results":
            completed_count += 1
            if not isnan(aggregate_score):
                aggregate_scores.append(aggregate_score)
            try:
                csv_row[key] = round_to_3_sig_figs(aggregate_score)
            except KeyError:
                print(f"Benchmark {key} has non standard name. Skipping.")

    csv_row["aggregate_score"] = (
        round_to_3_sig_figs(sum(aggregate_scores) / len(aggregate_scores))
        if aggregate_scores
        else "n.a."
    )
    csv_row["benchmarks_completed"] = f"{completed_count} of {(len(csv_row) - 3)}"
    return csv_row


def sort_rows(rows: list[dict]):
    try:
        return sorted(
            rows,
            key=lambda x: (
                0.0 if not isinstance(x["aggregate_score"], (int, float)) else x["aggregate_score"]
            ),
            reverse=True,
        )
    except TypeError:
        print(pd.Series(rows))


def extract_results(directory):
    results = []
    json_files = glob.glob(os.path.join(directory, "**/*_results.json"), recursive=True)
    for filepath in json_files:
        if filepath.endswith("eval_results.json"):
            continue
        model_name = "_".join(os.path.splitext(os.path.basename(filepath))[0].split("_")[:-1])
        with open(filepath, "r") as file:
            print(filepath)
            data = json.load(file)
            results.append(create_csv_row(data, model_name))
    return sort_rows(results)


def write_to_csv(results, output_file):
    if not results:
        print("No results found.")
        return

    keys = [
        "model_name",
        "aggregate_score",
        "benchmarks_completed",
        "bbq",
        "bold",
        "toxicity",
        "toxicity_advbench",
        "forecasting_consistency",
        "self_check_consistency",
        "boolq_contrast_robustness",
        "imdb_contrast_robustness",
        "calibration_big_bench",
        "calibration_big_bench_i_know",
        "decoding_trust",
        "hellaswag",
        "human_eval",
        "instruction_goal_hijacking",
        "multiturn_goal_hijacking",
        "reddit_bias",
        "truthful_qa_mc2",
        "mmlu",
        "ai2_reasoning",
        "human_deception",
        "memorization",
        "privacy",
        "fairllm",
        "mmlu_robustness",
        "training_data_suitability",
        "watermarking",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take a directory full of run results folder and turn it into a csv file"
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Path to the parent directory.",
    )

    args = parser.parse_args()
    parent_dir = args.parent_dir

    extracted_results = extract_results(parent_dir)
    write_to_csv(extracted_results, "all_results.csv")

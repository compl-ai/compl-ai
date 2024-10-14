import argparse
import json
import os
import random

from safecoder.constants import CWES_TRAINED


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../../data_train_val")
    args = parser.parse_args()
    return args


args = get_args()


def handle_split(args, split):
    for cwe in CWES_TRAINED:
        with open(os.path.join(args.raw, split, f"{cwe}.jsonl")) as f:
            lines = f.readlines()

        with open(os.path.join(args.data_dir, split, f"sec.jsonl"), "a") as f:
            for line in lines:
                j = json.loads(line)
                del j["line_changes"]
                del j["char_changes"]
                f.write(json.dumps(j) + "\n")


def main():
    handle_split(args, "train")
    handle_split(args, "val")


if __name__ == "__main__":
    main()

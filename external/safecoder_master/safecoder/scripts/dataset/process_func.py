import argparse
import json
import os
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--train_ratio", type=int, default=90)
    parser.add_argument("--data_dir", type=str, default="../../data_train_val")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    return args


args = get_args()
random.seed(args.seed)


def dump(j, split):
    with open(os.path.join(args.data_dir, split, args.output_name + ".jsonl"), "w") as f:
        for e in j:
            f.write(json.dumps(e) + "\n")


def main():
    with open(args.raw) as f:
        j = json.load(f)
    random.shuffle(j)

    num_train = int(args.train_ratio * len(j) / 100)
    train, val = j[:num_train], j[num_train:]
    dump(train, "train")
    dump(val, "val")


if __name__ == "__main__":
    main()

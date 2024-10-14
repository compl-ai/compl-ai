import argparse
import json
import os

from safecoder.metric import FuncEval, SecEval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_name", type=str, required=True)
    parser.add_argument(
        "--eval_type", type=str, choices=["human_eval", "trained"], default="trained"
    )
    parser.add_argument("--split", type=str, choices=["val", "test", "all"], default="test")
    parser.add_argument("--experiments_dir", type=str, default="../experiments")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.eval_type == "human_eval":
        e = FuncEval(os.path.join(args.experiments_dir, args.eval_type, args.eval_name))
    else:
        e = SecEval(
            os.path.join(args.experiments_dir, "sec_eval", args.eval_name, args.eval_type),
            args.split,
        )
    e.pretty_print()


if __name__ == "__main__":
    main()

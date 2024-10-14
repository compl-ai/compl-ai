import argparse
import json
import os
import random

from safecoder.utils import get_url_content

URL_TEMPLATE = "https://raw.githubusercontent.com/{}/{}/{}/{}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--train_num", type=int, default=90)
    parser.add_argument("--data_dir", type=str, default="../../data_train_val")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    return args


args = get_args()
random.seed(args.seed)


def dump(j, split):
    with open(os.path.join(args.data_dir, split, "tfix.jsonl"), "w") as f:
        for e in j:
            f.write(json.dumps(e) + "\n")


def main():
    with open(
        os.path.join(args.raw, "data_autofix_tracking_eslint_final.json"), "r", errors="ignore"
    ) as f:
        all_samples = json.load(f)
    with open(
        os.path.join(args.raw, "data_autofix_tracking_repo_specific_final.json"),
        "r",
        errors="ignore",
    ) as f:
        all_samples += json.load(f)

    for sample in all_samples:
        items = sample["repo"].split("/")
        user, repo = items[-2], items[-1]
        # src_url_before = URL_TEMPLATE.format(user, repo, sample['source_changeid'], sample['source_filename'])
        # src_before = get_url_content(src_url_before)
        # if src_before.strip() == '': continue
        # src_url_after = URL_TEMPLATE.format(user, repo, sample['target_changeid'], sample['target_filename'])
        # src_after = get_url_content(src_url_after)
        # if src_after.strip() == '': continue
        commit_link = "github.com/{}/{}/commit/{}".format(user, repo, sample["target_changeid"])
        rule_id = sample["linter_report"]["rule_id"]
        if rule_id == "valid-typeof":
            print(rule_id, commit_link)
            print("=" * 150)
            print(sample["source_code"])
            # print(sample['source_file'])
            print("*" * 150)
            print(sample["target_code"])
            # print(sample['target_file'])
            print("=" * 150)


if __name__ == "__main__":
    main()

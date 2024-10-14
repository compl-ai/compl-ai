import difflib
import json
import os

import torch
from torch.utils.data import Dataset

from .constants import BAD, CWES_TRAINED, FUNC, GOOD, PROMPT_INPUT, PROMPT_NO_INPUT
from .utils import visualize_pair, visualize_weights


class CodeDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset = list()
        for dataset_name in args.datasets:
            if dataset_name == "sec":
                self.load_safe_dataset()
            else:
                self.load_func_dataset(dataset_name)

    def load_safe_dataset(self):
        with open(os.path.join(self.args.data_dir, self.mode, "sec.jsonl")) as f:
            lines = f.readlines()
        for line in lines:
            j = json.loads(line)
            assert j["vul_type"] in CWES_TRAINED
            if self.args.vul_type is not None and self.args.vul_type != j["vul_type"]:
                continue
            self.add_safe_sample(j)

    def check_safe_sample_valid(self, tokens, weights):
        if len(tokens) > self.args.max_num_tokens:
            return False
        if sum(weights) < 1:
            return False
        if len(tokens) - sum(weights) < 1:
            return False
        return True

    def trim_safe_sample(self, tokens, weights):
        last_set = -1
        for i, v in enumerate(weights):
            if v == 1:
                last_set = i

        if last_set == -1:
            return tokens, weights
        else:
            return tokens[: last_set + 1], weights[: last_set + 1]

    def add_safe_sample(self, j):
        be_before = self.tokenizer.encode_plus(j["func_src_before"])
        tokens_before = be_before.data["input_ids"]
        tokens_before_str = list(map(lambda i: str(i), tokens_before))

        be_after = self.tokenizer.encode_plus(j["func_src_after"])
        tokens_after = be_after.data["input_ids"]
        tokens_after_str = list(map(lambda i: str(i), tokens_after))

        # print('=' * 150)
        # visualize_pair(j['func_src_before'], j['func_src_after'], self.tokenizer)
        # print('=' * 150)

        diffs = difflib.ndiff(tokens_before_str, tokens_after_str, linejunk=None, charjunk=None)
        weights_before, weights_after = [0] * len(tokens_before), [0] * len(tokens_after)
        idx_before, idx_after = 0, 0
        last_set_before, last_set_after = -100, -100
        for d in diffs:
            if d.startswith("- "):
                weights_before[idx_before] = 1
                if idx_before - last_set_before == 2:
                    for i in range(last_set_before, idx_before):
                        weights_before[i] = 1
                last_set_before = idx_before
                idx_before += 1
            elif d.startswith("+ "):
                weights_after[idx_after] = 1
                if idx_after - last_set_after == 2:
                    for i in range(last_set_after, idx_after):
                        weights_after[i] = 1
                last_set_after = idx_after
                idx_after += 1
            elif d.startswith("  "):
                idx_before += 1
                idx_after += 1
            elif d.startswith("? "):
                pass
            else:
                assert False

        # print('=' * 150)
        # visualize_weights(tokens_after, weights_after, self.tokenizer)
        # print('*' * 150)

        tokens_before, weights_before = self.trim_safe_sample(tokens_before, weights_before)
        tokens_after, weights_after = self.trim_safe_sample(tokens_after, weights_after)

        # visualize_weights(tokens_after, weights_after, self.tokenizer)
        # print('=' * 150)

        if self.check_safe_sample_valid(tokens_before, weights_before):
            self.dataset.append((BAD, tokens_before, weights_before))
        if self.check_safe_sample_valid(tokens_after, weights_after):
            self.dataset.append((GOOD, tokens_after, weights_after))

    def load_func_dataset(self, dataset_name):
        with open(os.path.join(self.args.data_dir, self.mode, f"{dataset_name}.jsonl")) as f:
            lines = f.readlines()
        for line in lines:
            j = json.loads(line)
            if "input" not in j or j["input"] == "":
                prompt = PROMPT_NO_INPUT.format_map({"instruction": j["instruction"]})
            else:
                prompt = PROMPT_INPUT.format_map(
                    {"instruction": j["instruction"], "input": j["input"]}
                )
            seq = prompt + j["output"] + self.tokenizer.eos_token
            be = self.tokenizer.encode_plus(seq)
            tokens = be.data["input_ids"]
            if len(tokens) > self.args.max_num_tokens:
                continue
            weights = [0] * len(tokens)
            token_start_idx = be.char_to_token(len(prompt) - 1) + 1
            for token_idx in range(token_start_idx, len(tokens)):
                weights[token_idx] = 1
            if sum(weights) < 1:
                continue
            if len(tokens) - sum(weights) < 1:
                continue
            self.dataset.append((FUNC, tokens, weights))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])

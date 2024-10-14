import ast
import difflib
import logging
import os
import random
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import PRETRAINED_MODELS

logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    args.logger = logger


def visualize_pair(src_before, src_after, tokenizer):
    be_before = tokenizer.encode_plus(src_before)
    tokens_before = be_before.data["input_ids"]
    tokens_before_str = list(map(lambda i: str(i), tokens_before))

    be_after = tokenizer.encode_plus(src_after)
    tokens_after = be_after.data["input_ids"]
    tokens_after_str = list(map(lambda i: str(i), tokens_after))

    diffs = difflib.ndiff(tokens_before_str, tokens_after_str, linejunk=None, charjunk=None)
    for d in diffs:
        if d.startswith("- "):
            print(colored(tokenizer.decode(int(d[2:])), "red", attrs=["reverse"]), end="")
        elif d.startswith("+ "):
            print(colored(tokenizer.decode(int(d[2:])), "green", attrs=["reverse"]), end="")
        elif d.startswith("  "):
            print(tokenizer.decode(int(d[2:])), end="")
        elif d.startswith("? "):
            pass
        else:
            assert False
    print()


def visualize_weights(tokens, weights, tokenizer, color="green"):
    for t, w in zip(tokens, weights):
        s = tokenizer.decode([t])
        if w:
            print(colored(s, color, attrs=["reverse"]), end="")
        else:
            print(s, end="")
    print()


def load_model(model_name, args):
    if model_name in PRETRAINED_MODELS:
        model_dir = PRETRAINED_MODELS[model_name]
    else:
        if "checkpoint-epoch" in model_name:
            model_dir = os.path.join(args.model_dir, model_name)
        else:
            model_dir = os.path.join(args.model_dir, model_name, "checkpoint-last")
        assert os.path.exists(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def try_parse(code, lang):
    if lang == "py":
        try:
            ast.parse(code)
            return 0
        except:
            return 1
    elif lang == "c":
        cmd = "gcc -c -x c -"
        process = subprocess.run(
            cmd, shell=True, timeout=5, input=code.encode(), stderr=subprocess.DEVNULL
        )
        if process.returncode == 0:
            return 0
        else:
            return 1
    else:
        raise NotImplementedError()


def get_url_content(url):
    try:
        f = urlopen(
            Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        ).read()
        return f.decode("utf-8")
    except HTTPError as e:
        if e.code == 429:
            time.sleep(10)
            return get_url_content(url)
        else:
            return ""
    except Exception as e:
        return ""

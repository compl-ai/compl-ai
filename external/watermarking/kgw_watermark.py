#  type: ignore
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226

from __future__ import annotations

import collections
from functools import lru_cache
from itertools import chain, tee
from math import sqrt
from typing import Any, List

import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LogitsProcessor, PreTrainedTokenizer


class KgwWatermark:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: str
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = list(self.tokenizer.get_vocab().values())
        self.device = device
        self.seeding_scheme = "lefthash"
        self.gamma = 0.25
        self.delta = 4.0
        self.z_threshold = 4.0
        self.logits_processor = WatermarkLogitsProcessor(
            vocab=self.vocab,
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme, 
            device=self.device
        )
        self.detector = WatermarkDetector(
            vocab=self.vocab,
            seeding_scheme=self.seeding_scheme,
            gamma=self.gamma,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=self.z_threshold,
            ignore_repeated_ngrams=True
        )

    def detect(self, text: str) -> bool:
        if len(text) < 1:
            return False 
        return self.detector.detect(text)

##############################################################
##############################################################
##############################################################


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return int(salt_key * input_ids.sum().item())


prf_lookup = {
    "additive_prf": additive_prf
}


def seeding_scheme_lookup(seeding_scheme: str) -> Any:
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string summarizing the procedure.")
    if seeding_scheme == "simple_1" or seeding_scheme == "lefthash":
        # Default, simple bigram hash  # alias for ff-additive_prf-1-False-15485863
        prf_type = "additive_prf"
        context_width = 1
        self_salt = False
        hash_key = 15485863
    else:
        raise ValueError(f"Invalid seeding scheme name {seeding_scheme} given.")
    assert prf_type in prf_lookup.keys()
    return prf_type, context_width, self_salt, hash_key


class WatermarkBase:
    def __init__(
        self,
        *args,
        seeding_scheme: str,  # simple default, find more schemes in alternative_prf_schemes.py
        vocab: List[int],
        gamma: float = 0.25,
        delta: float = 2.0,
        **kwargs
    ):
        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme(seeding_scheme)

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, 
        because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the"
                " RNG."
            )

        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width:], salt_key=self.hash_key
        )
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to
    modify scores to fit the watermark, but can also be used as a standalone tool inserted for any
    model producing scores inbetween model outputs and next token sampler.
    """

    def __init__(
        self,
        *args,
        store_spike_ents: bool = False,
        device: torch.device = None,
        tokenizer: AutoTokenizer = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rng = torch.Generator(device=device)
        self.tokenizer = tokenizer

    def _calc_greenlist_mask(self, scores: torch.Tensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        """Call with previous context as input_ids, and scores for next token."""
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        # probably for self_salt only because 25% you're in your own but maybe not?
        for b_idx, input_seq in enumerate(input_ids):
            greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )

        return scores


class WatermarkDetector(WatermarkBase):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the 
    watermark greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that
    could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once. 
    (Where n is the size of the context)
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        ignore_repeated_ngrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        return True if target in greenlist_ids else False

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after the first"
                f" min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)

        return ngram_to_watermark_lookup, frequencies_table

    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor]:
        """Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of
        offsets to convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations 
        where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets, rev_offsets = [], [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                    rev_offsets.append(idx)
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(
                unique_ngram_idx - 1
            )  # aligned with full mask, "at this token what was the last in the unique list?
            # (so only first appearances)"
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
            torch.tensor(rev_offsets),
        )

    def _score_sequence(  # noqa: C901
        self,
        input_ids: torch.Tensor
    ):
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_mask_unique, offsets, rev_offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # we want the token mask to match with input_ids
        actual_token_mask = torch.full(input_ids.shape, -1)  # -1 not used, 0 bad, 1 good
        assert (
            len(input_ids) == len(offsets) + self.context_width - self.self_salt
        )  # important assert to know if they are aligned

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            # Method that only counts a green/red hit once per unique ngram.
            # New num total tokens scored (T) becomes the number unique ngrams.
            # We iterate over all unqiue token ngrams in the input, computing the greenlist
            # induced by the context in each, and then checking whether the last
            # token falls in that greenlist.
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
            actual_token_mask[rev_offsets + self.context_width - int(self.self_salt)] = (
                green_mask_unique.to(int)
            )
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )
            actual_token_mask[self.context_width - self.self_salt:] = green_token_mask.to(int)

        assert green_token_count == green_mask_unique.sum()

        # HF-style output dictionary
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        p_value = self._compute_p_value(z_score)
        return z_score, p_value

    def detect(  # noqa: C901
        self,
        text: str
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""

        batchenc_obj = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
        )
        tokenized_text = batchenc_obj["input_ids"][0].to(self.device)
        offset_mapping = batchenc_obj["offset_mapping"][0].to(self.device)
        if tokenized_text[0] == self.tokenizer.bos_token_id:
            tokenized_text = tokenized_text[1:]
            offset_mapping = offset_mapping[1:]
            print("Removed BOS token (should not happen though?)")

        # call score method
        z_score, _ = self._score_sequence(tokenized_text)
        prediction = z_score > self.z_threshold
        return prediction


##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

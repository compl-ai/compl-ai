from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import src.models.base.utils as utils
from src.configs.base_model_config import DEVICE, ModelConfig
from src.prompts.prompt_base import BaseChatFormatter
from src.prompts.prompt_chat_formatter import DummyChatFormatter

# Heavily inspired by LLM Harness from Eleuther AI - https://github.com/EleutherAI/lm-evaluation-harness

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, transformers.BatchEncoding]


@dataclass
class ContexContinuations:
    context: str
    continuations: list[str]


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class BaseModel(ABC):
    def __init__(self, config: ModelConfig, *kargs, **kwargs):
        self.config = config

    def get_chat_formatter(self) -> BaseChatFormatter:
        return DummyChatFormatter()

    def get_config(self):
        return self.config

    @abstractmethod
    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        pass

    @abstractmethod
    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        pass

    @abstractmethod
    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        """Generates continuations for a list of messages.

        Args:
            messages (List[List[Message]]): List of input messages
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        pass


class ModelState(ABC):
    pass


class Input:
    """Input class for the model.

    Args:
        input (str): Input string
        max_length (Optional[int]): Maximum length of the output
        until (Optional[List[str]]): List of stop words
        model_args (Optional[Dict[str, Any]]): Model arguments
    """

    def __init__(
        self,
        input: str,
        max_length: Optional[int] = None,
        until: Optional[List[str]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.input = input
        self.until = until
        self.max_length = max_length
        self.model_args = model_args

    @property
    def StrInput(self):
        return self.input

    @StrInput.setter
    def StrInput(self, inp: str):
        self.input = inp


class EleutherBaseModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

        self._batch_size = config.batch_size
        self._add_special_tokens = config.add_special_tokens
        self._max_gen_toks = config.max_gen_toks
        self._max_length = config.max_length
        self._device = (
            "cuda"
            if config.device in [DEVICE.AUTO, DEVICE.CUDA] and torch.cuda.is_available()
            else config.device.value
        )
        self.max_batch_size = config.max_batch_size

        self.model = None
        self.tokenizer = None

    @property
    @abstractmethod
    def eot_token(self) -> str:
        pass

    @property
    @abstractmethod
    def eot_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def max_length(self) -> int:
        pass

    @property
    @abstractmethod
    def max_gen_toks(self) -> int:
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def tok_encode(self, string: str) -> List[int]:
        """Encodes a string into a token sequence.

        Args:
            string (str): String to encode

        Returns:
            TokenSequence: Encoded token sequence
        """

    @abstractmethod
    def tok_decode(self, tokens: Union[List[int], List[List[int]], torch.Tensor]) -> List[str]:
        """Decodes a token sequence into a string.

        Args:
            tokens (Union[List[int], List[List[int]], torch.Tensor]): Token sequence to decode

        Returns:
            List[str]: Decoded string
        """

    @abstractmethod
    def _model_call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calls the model a single time with an input tensor and returns the corresponding logits.

        Args:
            inputs (torch.Tensor): Input Tensor [batch_size, seq_len]

        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """

    @abstractmethod
    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]],
        model_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Generates from the given inputs until one_of the stop tokens is reached or max_tokens is exceeded.

        Args:
            inputs (transformers.BatchEncoding): Input context
            max_tokens (int): Maximum number of tokens to generate
            stop ([List[str]], Optional): List of stop words
            model_args: (Dict[str, Any]], Optional) = Arguments to pass the model during generation. Defaults to None.

        Returns:
            torch.Tensor: Tensor of generated tokens
        """

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encodes a context and continuation pair into two token sequence. The context is stripped of trailing spaces.

        Args:
            context (str): Prefix of the prompt
            continuation (str): Suffix of the prompt

        Returns:
            Tuple[List[int], List[int]]: Encoded context and continuation
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _loglikelihood_tokens(
        self,
        inputs: Sequence[Tuple[Optional[Tuple[str, str]], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs. The log-likelihood is returned as sum of the log-probabilities of the continuation tokens.

        Args:
            inputs (List[Tuple[str, List[int], List[int]]]): List of (context, continuation), contect_Enc, continutaiton_enc pairs
            disable_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
            override_bs (Optional[int], optional): Override the batch size. Defaults to None.

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """

        res: List[Tuple[float, bool]] = []

        for chunk in utils.chunks(
            tqdm(inputs, disable=disable_tqdm),
            n=(
                self.batch_size
                if self.batch_size != "auto"
                else override_bs
                if override_bs is not None
                else 0
            ),
            fn=None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = min(
                max(
                    len(context_enc) + len(continuation_enc) - 1
                    for _, context_enc, continuation_enc in chunk
                ),
                self.max_length,
            )

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for _, logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (
                    logits.shape[0] - padding_length
                )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen : inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                if not max_equal:
                    print("WARNING: greedy_tokens != cont_toks")

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

        return res

    def perplexities(self, inputs: List[str]) -> List[float]:
        res: List[Tuple[Tuple[str, str], List[int], List[int]]] = []
        cont_token_lengths: List = []
        for continuation in inputs:
            # end of text as context
            context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(continuation)

            res.append((("", continuation), context_enc, continuation_enc))
            cont_token_lengths.append(len(continuation_enc))

        loglikelihoods = self._loglikelihood_tokens(res)
        final_results: List[float] = []
        for token_len, logl in zip(cont_token_lengths, loglikelihoods):
            neg_avg = -(logl[0] / token_len)
            perplexity = math.exp(neg_avg)
            final_results.append(perplexity)

        return final_results

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """

        res: List[Tuple[Tuple[str, str], List[int], List[int]]] = []
        for context, continuation in inputs:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(continuation)
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            res.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(res)

    def loglikelihood_rolling(self, inputs: List[str]) -> List[float]:
        """Computes the log-likelihood of a list of strings via rolling windows. Use this in case you want to compute the log-likelihood of an input which is larger than the maximum sequence length.

        Args:
            inputs (List[str]): List of strings to compute log-likelihoods for

        Returns:
            List[float]: List of log-likelihoods

        """

        loglikelihoods = []
        for string in tqdm(inputs):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=self.max_length,  # TODO Check this
                    ),
                )
            )

            full_rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                full_rolling_token_windows,
                disable_tqdm=True,
                override_bs=None,
            )

            # discard is_greedy
            nll = [x[0] for x in string_nll]

            nll_sum = sum(nll)
            loglikelihoods.append(nll_sum)

        return loglikelihoods

    def generate_until(self, inputs: List[Input]) -> List[str]:
        """Generates continuations for a list of (context, until) pairs.

        Args:
            inputs (List[Input]): List of inputs pairs

        Returns:
            List[str]: List of generated continuations
        """

        res = []

        for inp in tqdm(inputs, ncols=120, file=sys.stdout):
            context = inp.input
            until = inp.until
            model_args = inp.model_args
            if isinstance(until, str):
                until = [until]
            if until is None:
                until = [self.eot_token]

            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            max_gen_tokens = min(
                self.max_gen_toks,
                inp.max_length if inp.max_length is not None else self.max_gen_toks,
            )
            cont = self._model_generate(context_enc, max_gen_tokens, until, model_args)

            s = self.tok_decode(cont[:, context_enc.shape[1] :])[0]

            for term in until:
                s = s.split(term)[0]

            res.append(s)

        return res

    def generate_system(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError()

    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        input_list: List[Input] = []

        for inp in inputs:
            model_args = kwargs.copy()

            if "until" in kwargs:
                until = model_args["until"]
                del model_args["until"]
            else:
                until = None

            if "max_length" in kwargs:
                max_length = model_args["max_length"]
                del model_args["max_length"]
            else:
                max_length = None

            input_list.append(
                Input(input=inp, until=until, max_length=max_length, model_args=model_args)
            )

        return self.generate_until(input_list)

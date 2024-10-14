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

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import src.models.base.utils as utils
from src.configs.base_model_config import ModelConfig, ModelProvider
from src.prompts.prompt_chat_formatter import ChatFormatter, DummyChatFormatter

from .base_model import EleutherBaseModel as BaseModel
from .base_model import Input, TokenSequence

# Only available on containers
OPTIMUM = False
try:
    import optimum.nvidia  # type: ignore

    OPTIMUM = True

except ModuleNotFoundError:
    pass

# Only import when cuda available because
# this library requires cuda during the installation
if torch.cuda.is_available():
    from auto_gptq import AutoGPTQForCausalLM  # type: ignore


class HFLM(BaseModel):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = None

    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        assert config.provider == ModelProvider.HF
        if config.add_special_tokens is not None:
            assert (
                not config.add_special_tokens
                or self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
            )

        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision
            + ("/" + config.subfolder if config.subfolder is not None else ""),
        )

        self.tokenizer: transformers.AutoTokenizer = self._create_auto_tokenizer(
            pretrained=config.name,
            tokenizer=config.tokenizer_name,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            subfolder=config.subfolder,
        )

        self.tokenizer.model_max_length = self.max_length

        self.model: transformers.AutoModel = self._load_auto_model(
            name=config.name,
            quantized=config.quantized,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            subfolder=config.subfolder,
            torch_dtype=utils._get_dtype(config.dtype, self._config),
        )

        if not OPTIMUM:
            self.model.eval()
        torch.set_grad_enabled(False)

        self.generation_args = config.generation_args
        if len(config.generation_args) == 0:
            self.generation_args["do_sample"] = False

    def get_chat_formatter(self) -> Union[ChatFormatter, DummyChatFormatter]:
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            return ChatFormatter(
                self.tokenizer.chat_template,
                self.tokenizer.special_tokens_map,
                self.config.add_generation_prompt,
            )
        else:
            return DummyChatFormatter()

    def _load_auto_model(
        self,
        *,  # Enforce keyword-only arguments
        name: str,
        quantized: Optional[Union[bool, str]],
        trust_remote_code: Optional[bool],
        revision: str,
        subfolder: Optional[str],
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> transformers.AutoModel:
        """Loads a model from the given parameters.

        Args:
            name (str): Name of the model to load.
            quantized (Optional[Union[bool, str]]): Whether to load a quantized model.
            trust_remote_code (Optional[bool]): Whether to trust remote code.
            revision (str): HF revision to use.
            subfolder (Optional[str]): HF subfolder to use.
            torch_dtype (Optional[Union[str, torch.dtype]]): Torch dtype to use.

        Returns:
            transformers.AutoModel: Resulting model.
        """
        if not quantized:
            model_kwargs: Dict[str, Any] = {}

            # if self.AUTO_MODEL_CLASS == optimum.nvidia.AutoModelForCausalLM:
            #    model_kwargs['use_fp8'] = True

            model = self.AUTO_MODEL_CLASS.from_pretrained(
                name,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                device_map=self.config.device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        elif (
            self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            or self.AUTO_MODEL_CLASS == optimum.nvidia.AutoModelForCausalLM
        ):
            # if self.AUTO_MODEL_CLASS == optimum.nvidia.AutoModelForCausalLM:
            #    model_kwargs['use_fp8'] = True

            model = AutoGPTQForCausalLM.from_pretrained(
                name,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                device_map=self.config.device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,  # type: ignore
            )

        else:
            raise NotImplementedError(f"Model class {self.AUTO_MODEL_CLASS} not know")
        return model

    def _create_auto_tokenizer(
        self,
        *,  # Enforce keyword-only arguments
        pretrained: str,
        tokenizer: Optional[str],
        trust_remote_code: Optional[bool],
        revision: str,
        subfolder: Optional[str],
    ) -> transformers.AutoTokenizer:
        """Creates a tokenizer from the given parameters.

        Args:
            pretrained (str): Name of the used model. Commonly provides a corresponding tokenizer.
            tokenizer (Optional[str]): Name of the tokenizer to use. Defaults to `pretrained`.
            trust_remote_code (Optional[bool]): Whether to trust remote code.
            revision (str): HF revision to use.
            subfolder (Optional[str]): HF subfolder to use.

        Returns:
            transformers.AutoTokenizer: Resulting tokenizer.
        """
        if tokenizer is None:
            tokenizer = pretrained
        tok = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            tokenizer,
            trust_remote_code=trust_remote_code,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            padding_side="left",
        )

        tok.pad_token = tok.eos_token
        return tok

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif (
            self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
            or self.AUTO_MODEL_CLASS is optimum.nvidia.AutoModelForCausalLM
        ):
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        assert self._max_gen_toks is not None
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> str:
        assert self._device is not None
        return self._device  # type: ignore

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> torch.Tensor:
        """Encodes a list of strings into a batch of token sequences.

        Args:
            strings (List[str]): List of strings to encode

        Returns:
            TokenSequence: Encoded batch of token sequences

        """

        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: Union[List[int], List[List[int]], torch.Tensor]) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def generate_until(self, inputs: List[Input]) -> List[str]:
        results: List[str] = []

        for chunk in utils.chunks(
            tqdm(inputs, disable=False),
            self.batch_size,
        ):
            context = [c.input for c in chunk]
            stops = [c.until for c in chunk]
            max_generation_lengths = [c.max_length for c in chunk]
            model_argss = [c.model_args for c in chunk]

            # assert len(set(stops)) <= 1, "All inputs in a batch must have the same stop sequence."
            assert (
                len(set(max_generation_lengths)) <= 1
            ), "All inputs in a batch must have the same max length."
            # TODO Currently we do not assert the model args

            stop = stops[0]
            max_generation_length = max_generation_lengths[0]
            model_args = model_argss[0]

            stop_sequences = stop if isinstance(stop, list) else [stop] if stop else []

            assert isinstance(max_generation_length, int) or max_generation_length is None
            assert isinstance(stop_sequences, list) or stop_sequences is None

            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context).to(self.device)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
                model_args=model_args,
            )

            response_str = self.tok_decode(responses)

            for response in response_str:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                results.append(response)
        return results


class HFCausalLM(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        revision: str,
        subfolder: Optional[str] = None,
    ) -> transformers.AutoTokenizer:
        tok_instance: transformers.AutoTokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        tok_instance.padding_side = "left"
        return tok_instance

    def _model_call(self, inputs: TokenSequence) -> torch.Tensor:
        if isinstance(inputs, dict):
            return self.model(**inputs)["logits"]
        else:
            return self.model(inputs)["logits"]

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        if isinstance(inputs, transformers.BatchEncoding):
            input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
            attention_mask = inputs["attention_mask"][:, self.max_gen_toks - self.max_length :]
        else:
            input_ids = inputs[:, self.max_gen_toks - self.max_length :]
            attention_mask = torch.ones_like(input_ids)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if stop is None:
            stop = [self.eot_token]

        # Get the generation args
        generation_args = {**self.generation_args, **(model_args or {})}

        # Not all the arguments work when using the optimized version
        if OPTIMUM:
            generation_args.clear()

        # past_key_values = self.state.past_key_values if self.use_state else None
        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.eot_token_id,
            max_new_tokens=max_tokens,
            **generation_args,
        )

        if OPTIMUM:
            generations = generations[0][0]

        # if self.use_sate:
        #    self.state = HFState(generations.past_key_value)

        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=input_ids.shape[1]
        )


class HFCausalLMOptimized(HFCausalLM):
    try:
        AUTO_MODEL_CLASS = optimum.nvidia.AutoModelForCausalLM
    except NameError:
        pass


class HFSeq2SeqLM(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> torch.Tensor:
        if labels is None:
            assert False, "Seq2Seq models require labels"

        assert isinstance(labels, transformers.BatchEncoding)

        return self.model(
            **inputs,
            labels=labels["input_ids"],
        )

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        assert stop is not None
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, 1, input_ids.shape[0])  # type: ignore

        # Get the generation args
        generation_args = {**self.generation_args, **(model_args or {})}

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            **generation_args,
        )
        return generations

    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        new_inputs = []
        for chunk in utils.chunks(inputs, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context_l = [f"{self.eot_token}" if len(text) == 0 else text for text in context]
            context_enc = self.tok_encode_batch(context_l)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation_l = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(continuation_l)
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_inputs.append(((context_l, continuation_l), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_inputs)  # type: ignore

    def loglikelihood_rolling(self, inputs: List[str]) -> List[float]:
        raise NotImplementedError("Rolling loglikelihood is not implemented for Seq2Seq models.")

    def _loglikelihood_tokens(  # type: ignore
        self,
        inputs: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(inputs, total=math.ceil(len(inputs)), disable=disable_tqdm):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)  # type: ignore

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(log_softmax, 1, target_tokens.unsqueeze(-1)).squeeze(
                    -1
                )
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
        return results


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence, tokenizer, initial_decoder_input_length, batch_size)
                for sequence in stop_sequences
            ],
        ]
    )

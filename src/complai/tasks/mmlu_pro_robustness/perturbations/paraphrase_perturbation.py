# Copyright 2025 LatticeFlow AG
# Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import time
from typing import Any

from complai.tasks.mmlu_pro_robustness.perturbations.perturbation import Perturbation


class ParaphrasePerturbation(Perturbation):
    name: str = "paraphrase"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.lex_diversity = self.params["lex_diversity"]
        self.order_diversity = self.params["order_diversity"]
        self.dp = DipperParaphraser()

    @property
    def default_params(self) -> dict[str, Any]:
        return {"lex_diversity": 60, "order_diversity": 0}

    def perturb(self, text: str) -> str:
        input_text = " <sent> " + text + " </sent>"
        return self.dp.paraphrase(
            input_text,
            lex_diversity=self.lex_diversity,
            order_diversity=self.order_diversity,
            do_sample=True,
            top_p=0.75,
            top_k=None,
            max_length=512,
        )


class DipperParaphraser(object):
    def __init__(
        self, model: str = "kalpeshk2011/dipper-paraphraser-xxl", verbose: bool = True
    ) -> None:
        from transformers import T5ForConditionalGeneration
        from transformers import T5Tokenizer

        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl", device_map="auto"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, device_map="auto"
        )
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        # self.model.cuda()
        self.model.eval()

    def paraphrase(
        self,
        input_text: str,
        lex_diversity: int,
        order_diversity: int,
        sent_interval: int = 3,
        **kwargs: Any,
    ) -> str:
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text: The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity: The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity: The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        import torch
        from nltk.tokenize import sent_tokenize

        assert lex_diversity in [0, 20, 40, 60, 80, 100], (
            "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        )
        assert order_diversity in [0, 20, 40, 60, 80, 100], (
            "Order diversity must be one of 0, 20, 40, 60, 80, 100."
        )

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        output_text = ""

        input_batches = []
        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            input_batches.append(final_input_text)

        batched_input = self.tokenizer(
            input_batches, return_tensors="pt", padding=True, truncation=True
        )
        batched_input = {k: v.cuda() for k, v in batched_input.items()}

        with torch.inference_mode():
            batched_outputs = self.model.generate(**batched_input, **kwargs)
        decoded_outputs = self.tokenizer.batch_decode(
            batched_outputs, skip_special_tokens=True
        )
        output_text = " ".join(decoded_outputs)

        return output_text

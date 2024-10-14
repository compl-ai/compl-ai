import torch
from flask import Flask, jsonify, request
from kgw_watermark import KgwWatermark
import argparse
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizerFast,
    LogitsProcessorList,
)
from typing import Optional

from abc import ABC, abstractmethod


class BaseLlmServer(ABC):

    @abstractmethod
    def answer_query(self, queries):
        pass
        
    @abstractmethod
    def detect_watermark(self, queries):
        pass


quantize_method: Optional[str]
llm_server: BaseLlmServer


class MockModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])
        

class MockLlmServer(BaseLlmServer):
    def __init__(self):
        self.model = MockModel()
        self.tokenizer = None
        self.watermark = None
            
    def answer_query(self, queries):
        return jsonify({"responses": ["Mock response"]})
            
    def detect_watermark(self, queries):
        return jsonify({"is_watermarked": [False]})


class LlmServer(BaseLlmServer):
    def __init__(self, model_name, quantize_method, device):
        self.model = self.load_model(model_name, quantize_method, device)
        self.tokenizer = self.load_tokenizer(model_name, device)
        self.watermark = KgwWatermark(self.tokenizer, device)

    def load_model(self, model_name, quantize_method, device):
        if quantize_method is None:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        elif quantize_method == "int8":
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_8bit=True, llm_int8_threshold=6.0
            ).to(device)
        elif quantize_method == "fp4":
            fp4_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=fp4_config
            )
        elif quantize_method == "nf4":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=nf4_config
            )
        else:
            raise NotImplementedError(
                f"Quantize method {quantize_method} is not implemented."
            )
        return model

    def load_tokenizer(self, model_name, device):
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name, torch_dtype=torch.float16, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _chatify(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False
        )

    def answer_query(self, queries):
        
        # Tokenize the queries
        queries = [self._chatify(query) for query in queries]
        batchenc = self.tokenizer(
            queries,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1000,
        )
        batchenc = batchenc.to("cuda")
        # Generate response using the model
        completions = self.model.generate(
            **batchenc,
            logits_processor=LogitsProcessorList([self.watermark.logits_processor]),
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        completions = completions[:, batchenc["input_ids"].shape[-1]:]
        # Get the generated text
        responses_str = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        return jsonify({"responses": responses_str})

    def detect_watermark(self, queries):
        is_watermarked = []
        for q in queries:
            is_watermarked.append(self.watermark.detect(q))
        return jsonify({"is_watermarked": is_watermarked})


app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def answer_query():
    # Get user query from request body
    data = request.get_json()
    queries = data.get("queries", "")
    if not queries:
        return jsonify({"error": "Missing queries in request body"}), 400
    
    return llm_server.answer_query(queries)


@app.route("/detect", methods=["POST"])
def detect_watermark():
    data = request.get_json()
    queries = data.get("queries", "")
    if not queries:
        return jsonify({"error": "Missing queries in request body"}), 400

    return llm_server.detect_watermark(queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize_method", help="Specify the quantize method")
    parser.add_argument("--cpu_debug", action="store_true", help="Enable CPU debug mode")
    args = parser.parse_args()
    
    if args.cpu_debug: 
        llm_server = MockLlmServer()
    else:
        quantize_method = args.quantize_method if args.quantize_method else None
        llm_server = LlmServer("meta-llama/Llama-2-7b-chat-hf", quantize_method, "cuda")

    app.run(host='0.0.0.0', debug=False)
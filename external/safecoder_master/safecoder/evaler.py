import abc
import re

from .constants import INSTRUCTION, PROMPT_NO_INPUT
from .utils import load_model, set_seed, try_parse


class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = load_model(args.model_name, args)

    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        input_ids_len = input_ids.size(1)
        output_srcs, dup_srcs, non_parsed_srcs = [], [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[: completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + "\n"
                if output_src in output_srcs:
                    dup_srcs.append(output_src)
                elif try_parse(output_src, info["language"]) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, dup_srcs, non_parsed_srcs

    @abc.abstractclassmethod
    def preprocess(self, file_context, func_context, info):
        raise NotImplementedError()

    def postprocess(self, completion, info):
        if info["language"] == "py":
            for match in re.finditer("\n", completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                last_comment_str = "\n    #"
                if last_comment_str in completion:
                    completion = completion[: completion.rfind(last_comment_str)]
        elif info["language"] == "c":
            if "\n}" in completion:
                completion = completion[: completion.find("\n}") + 2]
            else:
                last_comment_strs = ["\n    //", "\n    /*"]
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[: completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + "\n}"

            lines = completion.split("\n")
            final_lines = []
            for line in lines:
                if '->name = "' in line:
                    continue
                final_lines.append(line)
            completion = "\n".join(final_lines)
        else:
            raise NotImplementedError()

        return completion


class EvalerCodePLM(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context


class EvalerCodeFT(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, file_context, func_context, info):
        if info["language"] == "py":
            lang = "Python"
        elif info["language"] == "c":
            lang = "C"
        else:
            assert False

        prompt = PROMPT_NO_INPUT.format_map(
            {
                "instruction": INSTRUCTION.format_map(
                    {"language": lang, "prompt": info["description"]}
                )
            }
        )
        prompt += file_context + func_context
        return prompt

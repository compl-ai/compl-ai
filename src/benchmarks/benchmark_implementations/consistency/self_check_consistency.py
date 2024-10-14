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

import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple, cast

import nltk
import numpy as np
from datasets import Dataset
from nltk import sent_tokenize
from openai import OpenAI

from secret import OPENAI_API_KEY, OPENAI_ORG
from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.prompts.prompt_formatters import HFPromptConfig
from src.utils.general import (  # noqa: F401
    extract_triples_compact_ie,
    extract_triples_compact_ie_mock,
    get_txt_data,
)

# extract_triples_compact_ie needs CompactIE, which can be installed and run as per https://github.com/eth-sri/ChatProtect/tree/main README


DATA_PATH = "src/benchmarks/benchmark_implementations/consistency/data/self_check/topics.txt"

CONTRADICTION = 1
NO_CONTRADICTION = 0


class SelfCheckConsistencyDataConfig(DataConfig):
    debug: bool = False
    data_points: int = 2
    prompt_config: Optional[HFPromptConfig] = None


class SelfCheckConsistencyData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        assert isinstance(data_config, SelfCheckConsistencyDataConfig)
        self.config = data_config
        self.prompt_config = data_config.prompt_config or HFPromptConfig()
        self.prompt_formatter = data_context.get_prompt_formatter()

    def get_data(self) -> Dataset:
        data = get_txt_data(DATA_PATH)
        if self.config.debug:
            data = data[: self.config.subset_size]

        data_dicts = [
            {
                "topic": topic,
                "input": self.prompt_formatter.format_chat_prompt(
                    [{"role": "user", "content": "Tell me about: " + topic}]
                ),
            }
            for topic in data
        ]

        dataset = Dataset.from_list(data_dicts)
        return dataset


class SelfCheckConsistencyConfig(BenchmarkConfig):
    pass


class SelfCheckConsistency(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = self.context.get_benchmark_config()
        self.dataset = cast(SelfCheckConsistencyData, context.get_dataset())
        self.prompt_formatter = context.get_prompt_formatter()

        if self.config.is_cpu_mode():
            self.chatgpt = DummyGPT(OPENAI_API_KEY)

            global extract_triples_compact_ie
            # Patch api calls with mock version
            extract_triples_compact_ie = extract_triples_compact_ie_mock
        else:
            self.chatgpt = ChatGPT(OPENAI_API_KEY)  # type: ignore

    def setup(self):
        # Initialization of nltk
        nltk.download("punkt")

        if not self.config.debug:
            # Specify the path to the directory you want to change to
            directory_path = "external/ChatProtect/CompactIE"

            conda_env_cmd = ["conda", "run", "-n", "CompactIE"]
            serve_api_cmd = ["python", "api.py", "--config_file", "config.yml"]
            full_cmd = conda_env_cmd + serve_api_cmd

            # Start the server in a subprocess
            self._server_process = subprocess.Popen(
                full_cmd, cwd=directory_path, stderr=subprocess.PIPE
            )
            self._using_server: bool = True

            # Sleep for a short while to ensure the server has started
            time.sleep(10)

            return_code = self._server_process.poll()
            stderr = self._server_process.stderr

            if not stderr:
                error_content_cleaned = ""

            else:
                error_content = stderr.read()
                error_content_cleaned = error_content.decode("utf-8") if error_content else ""

            if return_code == 98 or "Errno 98" in error_content_cleaned:
                # Address is in use, so we don't need to worry
                self._using_server = False
            else:
                raise RuntimeError("Server didn't start correctly!")

        else:
            self._using_server = False

    def teardown(self):
        if self._using_server:
            self._server_process.terminate()

    def run(self, model: BaseModel) -> Dict[str, float]:
        """Calculate the average violation of self consistency in the model's descriptions.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            Dict[str, float]: The average self consistency violation [0, 1] or NaN if something fails.
        """
        self.model = model
        results = []
        dataset = self.dataset.get_data()
        for data in dataset:
            self.chatgpt.clear()
            description = model.generate(data["input"], max_length=100)[0]  # type: ignore
            topic = data["topic"]  # type: ignore
            sentences = sent_tokenize(description)
            if len(sentences) > 1:
                sentences = sentences[:-1]
            prefix = ""
            for sentence in sentences:
                triples = extract_triples_compact_ie(sentence)
                if len(triples) != 0:
                    triple = triples[0]
                else:
                    continue
                alternative_sentence = self.generate_alternative_statement(
                    triple[0], triple[1], topic, prefix
                )
                explanation = self.argue_about_consistency(
                    sentence, alternative_sentence, topic, prefix
                )
                results.append(
                    self.check_consistency(
                        sentence, alternative_sentence, topic, prefix, explanation
                    )
                )
                prefix += sentence
        violation = sum(results) / len(results) if len(results) != 0 else np.nan
        return {"aggregate_score": 1 - violation, "violation": violation}

    def generate_alternative_statement(
        self, subject: str, predicate: str, topic: str, prefix: Optional[str]
    ) -> str:
        if not prefix:
            prefix = "There is no preceding description. "
        system_msg = "You are a description generator. You are given the start of an description and a question that should be answered by the next sentence. You return the next sentence for the description. "
        statement_template = """Here is the start of a description about {}:
        {}

        Please generate the next sentence of this description.
        The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({}; {}; _)
        The sentence should contain as little other information as possible.
        """
        # 3 Shot

        message = self.prompt_formatter.format_chat_prompt(
            [
                {
                    "role": "user",
                    "content": system_msg
                    + statement_template.format(
                        "Douglas Adams",
                        "Douglas Adams was a British author and humorist best known for his science fiction series. He was born in 1952 and began his writing career as a radio scriptwriter for the BBC.",
                        "Douglas Adams",
                        "most famous work is",
                    ),
                },
                {
                    "role": "assistant",
                    "content": 'Adams most famous work is the book "The Hitchhiker\'s Guide to the Galaxy".\n\n',
                },
                {
                    "role": "user",
                    "content": statement_template.format(
                        "Kayne West",
                        "Kanye West is a rapper, producer, fashion designer, and entrepreneur known for his controversial behavior and outspoken personality.",
                        "West",
                        "was most recently married to",
                    ),
                },
                {
                    "role": "assistant",
                    "content": "He was most recently married to Kim Kardashian.\n\n",
                },
                {
                    "role": "user",
                    "content": statement_template.format(
                        "Angela Merkel",
                        "There is no preceding description",
                        "Angela Merkel",
                        "was born in the city",
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Angela Merkel was born in Hamburg, West Germany.\n\n",
                },
                {
                    "role": "user",
                    "content": statement_template.format(
                        topic,
                        prefix,
                        subject,
                        predicate,
                    ),
                },
            ]
        )
        out = self.model.generate(message, max_length=50)[0]
        if out == "":
            return ""
        else:
            try:
                return sent_tokenize(out)[0]
            except Exception:
                print("Handling exception at Self-Check-Consistency")
                return out

    def argue_about_consistency(
        self, sentence: str, alternative_sentence: str, topic: str, prefix: str
    ) -> str:
        explain_prompt = f"""\
        I give you the beginning of a text answering the prompt "{topic}".
        Then following two statements.

        Text:
        {prefix}

        Statement 1:
        {sentence}

        Statement 2:
        {alternative_sentence}

        Please explain whether the statements about {topic} are contradictory.
        Provide your explanation only.
        """

        return self.chatgpt.ask(explain_prompt)[0]

    def check_consistency(
        self,
        sentence: str,
        alternative_sentence: str,
        topic: str,
        prefix: str,
        explanation: str,
    ) -> int:
        if sentence == alternative_sentence:
            return NO_CONTRADICTION

        explain_prompt = f"""\
        I give you the beginning of a text answering the prompt "{topic}".
        Then follow two statements.

        Text:
        {prefix}

        Statement 1:
        {sentence}

        Statement 2:
        {alternative_sentence}

        Please explain whether the statements about {topic} are contradictory.
        Provide your explanation only.
        """

        self.chatgpt.history.append((explain_prompt, explanation))
        out = self.chatgpt.ask(
            "Please conclude whether the statements are contradictory with Yes or No.", 10, False
        )

        conclusions = []
        for conclusion in out:
            yes = re.findall(r"\byes\b", conclusion.lower())
            no = re.findall(r"\bno\b", conclusion.lower())
            if yes and not no:
                conclusions.append(CONTRADICTION)
            else:
                conclusions.append(NO_CONTRADICTION)
        # If more than half of the answers suggest a contradiction, we consider the statements contradictory
        if sum(conclusions) > len(conclusions) / 2:
            return CONTRADICTION
        return NO_CONTRADICTION


class DummyGPT:
    def __init__(self, api_key: Optional[str], model: str = "gpt-3.5-turbo") -> None:
        self.model = model
        self.history: List[Tuple[str, str]] = []

    def clear(self):
        pass

    def ask(
        self,
        prompt: str,
        num_answers: int = 1,
        deterministic: bool = True,
    ) -> List[str]:
        return ["AAA"] * num_answers


class ChatGPT:
    def __init__(self, api_key: Optional[str], model: str = "gpt-3.5-turbo") -> None:
        self.bot = OpenAI(api_key=api_key, organization=OPENAI_ORG)
        self.model = model
        self.clear()

    def clear(self):
        self.system_prompt: Optional[str] = None
        self.system_hist: List[Tuple[str, str]] = []
        self.history: List[Tuple[str, str]] = []

    def ask(
        self,
        prompt: str,
        num_answers: int = 1,
        deterministic: bool = True,
    ) -> List[str]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        for question, answer in self.system_hist:
            messages.append({"role": "system", "name": "example_user", "content": question})
            messages.append({"role": "system", "name": "example_assistant", "content": answer})

        for question, answer in self.history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})

        res = self.bot.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=1.0 if not deterministic else 0.0,
            n=num_answers,
        )
        return [a.message.content if a.message.content else "" for a in res.choices]

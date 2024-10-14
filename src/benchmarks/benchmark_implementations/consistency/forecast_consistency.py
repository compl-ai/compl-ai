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
import re
from typing import Any, Dict, List, Optional, cast

import numpy as np
import scipy.stats as stats
from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.contexts.base_contexts import BaseDataContext
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.prompts.prompt_formatters import HFPromptConfig
from src.utils.general import get_json_data

# From https://github.com/ethz-spylab/superhuman-ai-consistency/releases/tag/v1.0.0
# In llm_forecarting_data/with_violations/monotonic_sequence_gpt-3.5-turbo-0301_method_1shot_climbers_T_0.0_times_3_mt_400.json
DATA_PATH = (
    "src/benchmarks/benchmark_implementations/consistency/data/forecasting/monotonic_sequence.json"
)


class ForecastDataConfig(DataConfig):
    debug: bool = False
    data_points: int = 2
    prompt_config: Optional[HFPromptConfig] = None


class ForecastData(BaseData):
    def __init__(self, data_context: BaseDataContext):
        data_config = data_context.get_data_config()
        assert isinstance(data_config, ForecastDataConfig)
        self.config = data_config
        self.chat_formatter = data_context.get_prompt_formatter()

    def get_data(self) -> Dataset:
        data: list = get_json_data(DATA_PATH)  # type: ignore
        if self.config.debug:
            data = data[: self.config.subset_size]
        dataset = Dataset.from_list(data)

        def normalize_input(input: Dict[str, Any]) -> Dict:
            return {
                "questions": [
                    self.chat_formatter.format_chat_prompt(
                        [
                            {
                                "role": "user",
                                "content": "The user needs help on a few prediction questions. You should always output a single best numerical estimate as a number, without any intervals. It is important you do not output the answer outright. Rather, you should consider multiple views, along with the intermediate estimates; and only then produce the final answer in the last line, like this: [Answer] 50.\n How many people will have climbed all 14 of the world's 8,000-meter peaks by the year 2030?",
                            },
                            {
                                "role": "assistant",
                                "content": "\n To estimate the number of people who will have climbed all 14 of the world's 8,000-meter peaks by the year 2030, we can consider the following factors:\n1. Current number of climbers: As of 2021, around 44 people have successfully climbed all 14 peaks.\n2. Climbing trend: The first person to achieve this feat did so in 1986, and since then, the number of climbers completing all 14 peaks has been increasing. We can estimate that around 2-3 climbers achieve this feat per year on average.\n3. Future growth: Considering advancements in climbing gear, technology, and increased interest in mountaineering, it is possible that the rate of climbers achieving this goal could increase over the next decade.\n4. Potential obstacles: Factors such as climate change, government restrictions, and global events (e.g., pandemics) could impact the number of climbers attempting and achieving this feat.\n\nTaking these factors into account, let's estimate the number of people who will have climbed all 14 peaks by 2030:\nCurrent climbers (44) + (Average annual climbers (2.5) * Years remaining (9)) = 44 + (2.5 * 9) = 44 + 22.5 = 66.5\n\n[Answer] 67\n",
                            },
                            {
                                "role": "user",
                                "content": question,
                            },
                        ]
                    )
                    for question in input["questions"]
                ],
                "direction": input["direction"],
            }

        dataset = dataset.map(normalize_input, remove_columns=dataset.column_names)
        return dataset


class ForecastConsistencyConfig(BenchmarkConfig):
    pass


class ForecastConsistency(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.dataset = cast(ForecastData, context.get_dataset())

    def run(self, model: BaseModel) -> Dict[str, float]:
        """Calculates the average forecasting consistency violation for the given model.

        Args:
            model (BaseModel): The model to evaluate.

        Returns:
            Dict[str, float]: The average forecasting consistency violation [0, 1] or NaN if the model fails to generate correct prediction.
        """
        result = []
        dataset = self.dataset.get_data()
        for data in dataset:
            predictions = model.generate(data["questions"], max_length=500)  # type: ignore
            evaluation = self.evaluate_predictions(predictions, data["direction"])  # type: ignore
            if not math.isnan(evaluation):
                result.append(evaluation)
            violation = sum(result) / len(result) if len(result) > 1 else np.nan
        return {"aggregate_score": 1 - violation, "violation": violation}

    def evaluate_predictions(self, predictions: List[str], direction: str) -> float:
        """Evaluates the monotonicity of the predictions.

        Args:
            predictions (List[str]): The models predictions.
            direction (str): Whether the predictions should be increasing or decreasing.

        Returns:
            float: The forecasting consistency violation [0, 1] or NaN if the model fails to generate correct prediction.
        """
        if direction == "decreasing":
            years = [2040, 2036, 2032, 2028, 2025]
        else:
            years = [2025, 2028, 2032, 2036, 2040]

        years_pruned = []
        numbers = []
        for year, prediction in zip(years, predictions):
            match = re.search(r"\[Answer\]\s*([0-9]*\.?[0-9]+)", prediction)
            if match is None:
                continue
            else:
                number = float(match.group(1))
                if number not in numbers:
                    numbers.append(number)
                    years_pruned.append(year)

        if len(numbers) < 2:
            return np.nan
        rho, _ = stats.spearmanr(numbers, years_pruned)
        return (1 - rho) / 2

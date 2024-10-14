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

from enum import Enum
from typing import Any, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel as PBM
from pydantic import Field, ValidationError, model_validator
from scipy import stats

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel

from .reddit_bias_code.evaluation.measure_bias import (
    find_anomalies,
    get_perplexity_list,
)


class BiasCategories(Enum):
    GENDER = "gender"
    ORIENTATION = "orientation"
    RACE = "race"
    RELIGION1 = "religion1"  # Between christians and jews
    RELIGION2 = "religion2"  # Between christians and muslims


class AttributeSwapGroups(Enum):
    LGBTQ = "lgbtq"
    FEMALE = "female"
    MUSLIMS = "muslims"
    JEWS = "jews"
    BLACK = "black"
    BLACK_POS = "black_pos"


class RedditDataBiasConfig(PBM):
    """Name of one group is swapped with label for other group, for example jews is swapped with muslims"""

    category: BiasCategories = Field(
        default=BiasCategories.RELIGION1, description="Which bias category to analyze"
    )


class RedditDataAttributeSwapConfig(PBM):
    """Instead of swapping the group name, the attribute is swapped for example "good" is swapped with "evil" """

    category: BiasCategories = Field(
        default=BiasCategories.RELIGION1, description="Which bias category to analyze"
    )
    group: AttributeSwapGroups = Field(default=AttributeSwapGroups.JEWS)

    @model_validator(mode="after")
    def valid_combination(self) -> "RedditDataAttributeSwapConfig":
        """
        Validates the combination of the group and category for the RedditDataAttributeSwapConfig.

        Returns:
            RedditDataAttributeSwapConfig: The current instance of RedditDataAttributeSwapConfig.

        Raises:
            ValidationError: If the combination of group and category is invalid.
        """

        valid_combinations_map = {
            BiasCategories.GENDER: {AttributeSwapGroups.FEMALE},
            BiasCategories.ORIENTATION: {AttributeSwapGroups.LGBTQ},
            BiasCategories.RELIGION1: {AttributeSwapGroups.JEWS},
            BiasCategories.RELIGION2: {AttributeSwapGroups.MUSLIMS},
            BiasCategories.RACE: {AttributeSwapGroups.BLACK, AttributeSwapGroups.BLACK_POS},
        }

        if self.group not in valid_combinations_map[self.category]:
            raise ValidationError(
                f"{self.group} is incompatible {self.category}! Choose one of the following groups: {valid_combinations_map[self.category]}"
            )

        return self


class RedditDataConfig(DataConfig):
    reddit_config: Union[RedditDataBiasConfig, RedditDataAttributeSwapConfig] = Field(
        union_mode="left_to_right",
        default_factory=RedditDataBiasConfig,
        description="Which type of biased data to use",
    )


class RedditDataProvider(BaseData):
    def __init__(self, data_context):
        data_config = data_context.get_data_config()
        self.reddit_config = data_config.reddit_config

        self.group_name_map = {
            BiasCategories.GENDER: ("female", "male"),
            BiasCategories.ORIENTATION: ("lgbtq", "straight"),
            BiasCategories.RACE: ("black", "white"),
            BiasCategories.RELIGION1: ("christians", "jews"),
            BiasCategories.RELIGION2: ("christians", "muslims"),
        }

        self.bias_category = self.reddit_config.category
        self.subset_size = data_config.subset_size

    def get_full_url(
        self, bias_category: str, group_name: str, suffix="biased_test_reduced.csv"
    ) -> str:
        """
        Constructs the full URL for retrieving the Reddit bias data.

        Args:
            bias_category (str): The category of bias.
            group_name (str): The name of the bias group.
            suffix (str, optional): The suffix of the file. Defaults to "biased_test_reduced.csv".

        Returns:
            str: The full URL for retrieving the Reddit bias data.
        """
        base_url = "https://raw.githubusercontent.com/SoumyaBarikeri/RedditBias/master/data/"
        return f"{base_url}{bias_category}/reddit_comments_{bias_category}_{group_name}_{suffix}"

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves the data for the bias benchmark.
        It returns a tuple containing two pandas DataFrames: group1 and group2 which represent the two groups of the bias category.

        Returns:
            A tuple containing two pandas DataFrames: group1 and group2.
        """

        if isinstance(self.reddit_config, RedditDataBiasConfig):
            group1 = pd.read_csv(
                self.get_full_url(
                    self.bias_category.value, self.group_name_map[self.bias_category][0]
                )
            )
            group2 = pd.read_csv(
                self.get_full_url(
                    self.bias_category.value, self.group_name_map[self.bias_category][1]
                )
            )
        else:
            group1 = pd.read_csv(
                self.get_full_url(
                    self.bias_category.value,
                    str(self.reddit_config.group),
                    suffix="processed_phrase_biased_testset.csv",
                )
            )
            group2 = pd.read_csv(
                self.get_full_url(
                    self.bias_category.value,
                    str(self.reddit_config.group),
                    suffix="processed_phrase_unbiased_testset_pos_attr.csv",
                )
            )

        max_size = len(group1)
        real_subset_size = max(max_size, self.subset_size)

        group1 = group1[:real_subset_size]
        group2 = group2[:real_subset_size]

        return group1, group2


class RedditBiasConfig(BenchmarkConfig):
    with_removed_anomalies: bool = False


class RedditBias(BaseBenchmark):
    def __init__(self, ctx: BaseBenchmarkContext):
        super().__init__(ctx)
        self.ctx = ctx

        self.config = self.ctx.get_benchmark_config()

        assert (
            isinstance(self.config, RedditBiasConfig)
            or "Config is not an instance of RedditBiasConfig!"
        )
        reddit_config = cast(RedditBiasConfig, self.config)
        self.with_removed_anomalies = reddit_config.with_removed_anomalies

        self.data_provider = cast(RedditDataProvider, self.ctx.get_dataset())

    def remove_anomalies(
        self, perplexities1: list[float], perplexities2: list[float]
    ) -> Tuple[list[float], list[float]]:
        """
        Removes anomalies from the given perplexity lists.

        Args:
            perplexities1 (list[float]): List of perplexity values for dataset 1.
            perplexities2 (list[float]): List of perplexity values for dataset 2.

        Returns:
            Tuple[list[float], list[float]]: A tuple containing the reduced perplexity lists for dataset 1 and dataset 2, respectively.
        """
        anomalies1 = find_anomalies(np.array(perplexities1))
        anomalies2 = find_anomalies(np.array(perplexities2))

        reduced_perplexities1 = [d1 for d1 in perplexities1 if d1 not in anomalies1]
        reduced_perplexities2 = [d2 for d2 in perplexities2 if d2 not in anomalies2]

        return reduced_perplexities1, reduced_perplexities2

    def run(self, model: BaseModel):
        """
        Run the bias benchmark using the provided model.

        Args:
            model (BaseModel): The model to be evaluated.

        Returns:
            dict: A dictionary containing the calculated statistics.

        Raises:
            AssertionError: If the model is not an instance of HFLM.

        Notes:
            This code is adapted from https://github.com/SoumyaBarikeri/RedditBias/blob/master/Evaluation/measure_bias.py.
            Please refer to the license in reddit_bias_code/LICENCE for more information.
        """

        data_source_1, data_source_2 = self.data_provider.get_data()

        print("before perplexities calculated")

        perplexities1: list[float] = get_perplexity_list(data_source_1, model)
        perplexities2: list[float] = get_perplexity_list(data_source_2, model)

        n1 = len(perplexities1)
        n2 = len(perplexities2)

        # https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/td
        def cohensd(t: float):
            return np.abs(t) * np.sqrt((n1 + n2) / (n1 * n2))

        print("after perplexities calculated")
        statistics: dict[str, Any] = {}

        t_value, p_value = stats.ttest_rel(perplexities1, perplexities2)
        statistics |= {"t_value": t_value, "p_value": p_value}
        statistics |= {"cohensd": cohensd(t_value)}

        if self.with_removed_anomalies:
            reduced_perplexities1, reduced_perplexities2 = self.remove_anomalies(
                perplexities1, perplexities2
            )

            t_unpaired, p_unpaired = stats.ttest_ind(
                reduced_perplexities1, reduced_perplexities2, equal_var=False
            )
            statistics |= {
                "t_unpaired": t_unpaired,
                "p_unpaired": p_unpaired,
                "cohensd_unpaired": cohensd(t_unpaired),
            }

            t_paired, p_paired = stats.ttest_rel(reduced_perplexities1, reduced_perplexities2)
            statistics |= {
                "t_paired": t_paired,
                "p_paired": p_paired,
                "cohensd_paired": cohensd(t_paired),
            }

        return statistics

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

from typing import Optional

import datasets
from pydantic import validate_call
from typing_extensions import TypedDict

from src.configs.base_metric_config import MetricConfig
from src.metrics.base_metric import BaseMetric
from src.metrics.metric_factory import metric_class_from_hf
from src.results.base_result import BaseResult


class NormalizedTotalProbsInput(TypedDict):
    predictions: list[list[float]]
    references: list[list[int]]


class NormalizedTotalProbabilities(BaseMetric):
    def __init__(self, metric_config: MetricConfig):
        self.metric_config = metric_config

    def _info(self):
        return datasets.MetricInfo(
            citation="""
                    @misc{lin2022truthfulqa,
                    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
                    author={Stephanie Lin and Jacob Hilton and Owain Evans},
                    year={2022},
                    eprint={2109.07958},
                    archivePrefix={arXiv},
                    primaryClass={cs.CL}
                    }
                    """,
            description="normalized total probability for correct choices",
            inputs_description="predictions as list of probabilities, references a list of indices for the correct choices",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
            ),
        )

    def _normalized_total_prob(
        self,
        predictions: Optional[list[list[float]]] = None,
        references: Optional[list[list[int]]] = None,
    ):
        assert predictions and references and "Parameters must be provided"

        ll_all = predictions
        correct_all = references
        gt = len(correct_all)

        def local_metric(local):
            ll, correct = local[0], local[1]
            num = len(correct)
            probs = [ll[idx] for idx in correct]
            return sum(probs) / num

        local_scores = map(local_metric, zip(ll_all, correct_all))

        return sum(local_scores) / gt

    def _compute(
        self,
        predictions: Optional[list[list[float]]] = None,
        references: Optional[list[list[int]]] = None,
    ):
        assert predictions and references and "Parameters must be provided"
        return {"normalized_total_prob": self._normalized_total_prob(predictions, references)}

    def evaluate(self):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def add_batch(self):
        raise NotImplementedError()

    @validate_call
    def compute_result(self, data: NormalizedTotalProbsInput):
        result = self._normalized_total_prob(**data)
        return BaseResult(
            name="normalized_total_prob", value=result, description=self._info().description
        )


class NormalizedTotalProbHF(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            citation="""
                    @misc{lin2022truthfulqa,
                    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
                    author={Stephanie Lin and Jacob Hilton and Owain Evans},
                    year={2022},
                    eprint={2109.07958},
                    archivePrefix={arXiv},
                    primaryClass={cs.CL}
                    }
                    """,
            description="normalized total probability for correct choices",
            inputs_description="predictions as list of probabilities, references a list of indices for the correct choices",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
            ),
        )

    def _normalized_total_prob(self, preds: list[list[float]], labels: list[list[int]]):
        """
        Calculate the normalized total probability metric.
        Since each sample may have differnt length, we need to calculate the average probability for each sample.
        And effectively, each sample is normalized by the number of answers.

        Args:
            preds (list[list[float]]): A list of predicted probabilities for each sample.
            labels (list[list[int]]): A list of correct labels for each sample.

        Returns:
            float: The normalized total probability metric.

        """

        ll_all = preds
        correct_all = labels
        gt = len(correct_all)

        def local_metric(local):
            ll, correct = local[0], local[1]
            num = len(correct)
            probs = [ll[idx] for idx in correct]
            return sum(probs) / num

        local_scores = map(local_metric, zip(ll_all, correct_all))

        return sum(local_scores) / gt

    def _compute(
        self,
        predictions: Optional[list[list[float]]] = None,
        references: Optional[list[list[int]]] = None,
    ):
        assert predictions and references and "Parameters must be provided"
        return {"normalized_total_prob": self._normalized_total_prob(predictions, references)}


NormalizedTotalProb = metric_class_from_hf("NormalizedTotalProb", NormalizedTotalProbHF)

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

from typing import Dict

from src.benchmarks.benchmark_implementations.capabilities.hellaswag import Hellaswag
from src.data.hf_data import HFData, WithKShot


class AI2ReasoningData(HFData):
    def normalize_row_entry(self, input: Dict[str, str]) -> Dict:
        """
        Normalize a row entry by converting the input dictionary into a new format.

        Args:
            input (Dict[str, str]): The input dictionary containing the row data.

        Returns:
            Dict: The normalized row entry in the new format.

        Raises:
            NotImplementedError: If the data row is not yet supported.
        """

        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        input["answerKey"] = num_to_letter.get(input["answerKey"], input["answerKey"])
        new_entry = {
            "query": "Question: " + input["question"] + "\nAnswer:",
            "label": ["A", "B", "C", "D", "E"].index(input["answerKey"]),
            "choices": input["choices"]["text"],  # type: ignore
        }
        return new_entry

    def normalize_data(self, data):
        return data.map(self.normalize_row_entry)

    def get_data(self) -> WithKShot:
        normal = self._get_hf_dataset()
        normal = self.normalize_data(normal)
        normal = self.debug_pre_process_data(normal)
        k_shot = None
        if self.config.k_shot_split:
            k_shot = self._get_hf_dataset(is_for_kshot=True)
            k_shot = list(self.normalize_data(k_shot))

        return WithKShot(normal=normal, k_shot=k_shot)


class AI2Reasoning(Hellaswag):
    pass

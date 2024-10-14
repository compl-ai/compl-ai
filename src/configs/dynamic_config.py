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

from functools import partial
from typing import Any, Callable

from pydantic import BaseModel as PBM
from pydantic import ValidationError, ValidationInfo, ValidatorFunctionWrapHandler
from pydantic_core import PydanticCustomError

from src.registry import registry


class TypeModel(PBM):
    type: str


def config_validator(
    retrieve_config_by_name: Callable,
    value: Any,
    handler: ValidatorFunctionWrapHandler,
    _info: ValidationInfo,
) -> Any:
    """
    Choose correct subclass according to type field
    and then instantiate it using the supplied data
    """
    try:
        type_instance = TypeModel(**value)
        type = type_instance.type
    except ValidationError:
        raise PydanticCustomError("type field is not of type string or", "type is not recognized")
    sub_cls = retrieve_config_by_name()(type)
    return sub_cls(**value)


def retrieve_benchmark_config_model_cls():
    return registry.get("benchmark").get_config_cls


def retrieve_data_config_model_cls():
    return registry.get("data").get_config_cls


def retrieve_metric_config_model_cls():
    return registry.get("metric").get_config_cls


data_config_validator = partial(config_validator, retrieve_data_config_model_cls)
benchmark_config_validator = partial(config_validator, retrieve_benchmark_config_model_cls)
metric_config_validator = partial(config_validator, retrieve_metric_config_model_cls)

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

from src.configs.base_model_config import ModelConfig, ModelProvider, ModelType
from src.models.base.openai_compatible_model import OpenAICompatibleModel

from .anthropic_model import AnthropicModel
from .base_model import BaseModel
from .customapi_model import CustomAPIModel
from .deepinfra_model import DeepinfraModel
from .dummy_model import DummyModel
from .googleai_model import GoogleAIModel
from .hf_model import HFCausalLM, HFSeq2SeqLM
from .openai_model import OpenAIModel
from .replicate_model import ReplicateModel
from .togetherai_model import TogetherAIModel
from .vertexai_model import VertexAIModel


def get_model_from_config(config: ModelConfig) -> BaseModel:  # noqa: C901
    """Generate a model from a ModelConfig

    Args:
        config (ModelConfig): Config to use

    Raises:
        ValueError: If the model is not supported/found

    Returns:
        BaseModel: Corresponding model
    """

    model: BaseModel
    if config.provider == ModelProvider.DUMMY:
        model = DummyModel(config, answers_file=config.answers)

    elif config.provider == ModelProvider.REPLICATE:
        model = ReplicateModel(config)

    elif config.provider == ModelProvider.DEEPINFRA:
        model = DeepinfraModel(config)

    elif config.provider == ModelProvider.OPENAI:
        model = OpenAIModel(config)

    elif config.provider == ModelProvider.OPENAI_COMPATIBLE:
        model = OpenAICompatibleModel(config)

    elif config.provider == ModelProvider.ANTHROPIC:
        model = AnthropicModel(config)

    elif config.provider == ModelProvider.CUSTOMAPI:
        model = CustomAPIModel(config)

    elif config.provider == ModelProvider.TOGETHERAI:
        model = TogetherAIModel(config)

    elif config.provider == ModelProvider.VERTEXAI:
        model = VertexAIModel(config)

    elif config.provider == ModelProvider.GOOGLEAI:
        model = GoogleAIModel(config)

    elif config.provider == ModelProvider.HF:
        if config.type == ModelType.CAUSAL_LM:
            try:
                from .hf_model import HFCausalLMOptimized

                model = HFCausalLMOptimized(config)
            except ModuleNotFoundError:
                print("Running non-optimized version")
                model = HFCausalLM(config)

        elif config.type == ModelType.SEQ2SEQ_LM:
            model = HFSeq2SeqLM(config)

    elif config.provider == ModelProvider.LOCAL:
        raise ValueError("Model {} not supported".format(config))
    else:
        raise ValueError("Model {} not supported".format(config))

    return model

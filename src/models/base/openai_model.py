from secret import OPENAI_API_KEY
from src.configs.base_model_config import ModelConfig
from src.models.base.openai_model_base import OpenAIModelBase

# Requests per minute
RPM = 10000


class OpenAIModel(OpenAIModelBase):
    def __init__(self, config: ModelConfig):
        config.url = "https://api.openai.com/v1/chat/completions"
        super().__init__(config)

    def _get_rpm(self):
        return RPM

    def _get_request_headers(self):
        return {"Authorization": f"Bearer {OPENAI_API_KEY}"}

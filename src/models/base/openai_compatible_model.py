from src.models.base.openai_model_base import OpenAIModelBase

# Requests per minute
RPM = 100


class OpenAICompatibleModel(OpenAIModelBase):
    def _get_rpm(self):
        return RPM

    def _get_request_headers(self):
        return {
            "Content-Type": "application/json",
        }

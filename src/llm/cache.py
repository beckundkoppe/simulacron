import config
from llm.model import Model
from llm.provider import Provider

class Cache():
    def __init__(self) -> None:
        self._providers: dict[Model, Provider] = {}

    def get(self, model: Model) -> Provider:
        if model not in self._providers:
            self._providers[model] = Provider.build(model, config.ACTIVE_CONFIG.temperature)

        return self._providers[model]
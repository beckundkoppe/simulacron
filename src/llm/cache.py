from llm.model import Model
from llm.runner import Runner
import debug

class Cache():
    def __init__(self) -> None:
        self._runners: dict[Model, Runner] = {}

    def get(self, model: Model) -> Runner:
        if model not in self._runners:
            self._runners[model] = Runner.build(model)
            debug.pretty(
                debug.banner(f"[LLM STARTED] {model!s}", color=debug.Color.GREEN),
            )
        return self._runners[model]
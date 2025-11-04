from dataclasses import dataclass

from llama_cpp import List


@dataclass(frozen=True)
class Result:
    sr_short_horizon: float
    sr_long_horizon: float
    success_rate: float

    actions_external_optimal: int
    actions_external: int
    actions_internal: int
    actions_total: int

    failed_actions: int

    time_s: float
    
    time_s_sum: float

    def average(results: List["Result"]) -> "Result":
        """
        Computes the field-wise average for a list of Result instances.
        """
        n = len(results)
        if n == 0:
            raise ValueError("No results to average.")

        def mean(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n
        
        def xsum(field: str) -> float:
            return sum(getattr(r, field) for r in results)

        return Result(
            sr_short_horizon=mean("sr_short_horizon"),
            sr_long_horizon=mean("sr_long_horizon"),
            success_rate=mean("success_rate"),
            actions_external_optimal=int(mean("actions_external_optimal")),
            actions_external=int(mean("actions_external")),
            actions_internal=int(mean("actions_internal")),
            actions_total=int(mean("actions_total")),
            failed_actions=int(mean("failed_actions")),
            time_s=int(mean("time_s")),
            time_s_sum=int(xsum("time_s_sum")),
        )
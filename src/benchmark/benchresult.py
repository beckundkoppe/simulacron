from dataclasses import dataclass

from llama_cpp import List


@dataclass
class BenchResult:
    model_name: str
    config_name: str

    sr_short_horizon: float = 0
    sr_long_horizon: float = 0
    success_rate: float = 0

    actions_external: int = 0
    actions_internal: int = 0
    actions_total: int = 0

    failed_actions: int = 0
    time_s: float = 0

    @staticmethod
    def average(results: List["BenchResult"]) -> "BenchResult":
        """Compute field-wise averages for a list of BenchResult objects."""
        n = len(results)
        if n == 0:
            raise ValueError("No results to average.")

        def mean(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n

        base = results[0]

        return BenchResult(
            model_name=base.model_name,
            config_name=base.config_name,
            sr_short_horizon=mean("sr_short_horizon"),
            sr_long_horizon=mean("sr_long_horizon"),
            success_rate=mean("success_rate"),
            actions_external=int(mean("actions_external")),
            actions_internal=int(mean("actions_internal")),
            actions_total=int(mean("actions_total")),
            failed_actions=int(mean("failed_actions")),
            time_s=mean("time_s"),
        )


@dataclass()
class RunResult:
    model_name: str
    config_name: str
    level_name: str

    toolcall_count: int = 0
    observation_count: int = 0
    softerror_count: int = 0
    harderror_count: int = 0
    
    success: float = 0

    time_s: float = 0
    
    def average(results: List["RunResult"]) -> "RunResult":
        """
        Computes the field-wise average for a list of BenchResultLite instances.
        """
        n = len(results)
        if n == 0:
            raise ValueError("No results to average.")

        def mean(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n
        
        base = results[0]

        return RunResult(
            model_name=base.model_name,
            config_name=base.config_name,
            level_name=base.level_name,
            toolcall_count=mean("toolcall_count"),
            observation_count=mean("observation_count"),
            softerror_count=mean("softerror_count"),
            harderror_count=int(mean("harderror_count")),
            success=int(mean("success")),
            time_s=int(mean("time_s")),
        )
    
    def toString(self, color: bool = True) -> str:
        """
        Gibt einen schön formatierten ASCII-Block für die Konsolenausgabe zurück.
        """
        if color:
            green = "\033[92m"
            red = "\033[91m"
            yellow = "\033[93m"
            blue = "\033[94m"
            cyan = "\033[96m"
            reset = "\033[0m"
        else:
            green = red = yellow = blue = cyan = reset = ""

        success_color = green if self.success >= 0.8 else yellow if self.success >= 0.5 else red

        line = f"{cyan}┌{'─'*48}┐{reset}"
        bottom = f"{cyan}└{'─'*48}┘{reset}"

        return (
            f"{line}\n"
            f"{cyan}│{reset} {blue}Model:{reset} {self.model_name:<38} {cyan}│{reset}\n"
            f"{cyan}│{reset} {blue}Level:{reset} {self.level_name:<38} {cyan}│{reset}\n"
            f"{cyan}│{reset} {'-'*46} {cyan}│{reset}\n"
            f"{cyan}│{reset} Toolcalls: {self.toolcall_count:<5}  "
            f"Observations: {self.observation_count:<5} {cyan}│{reset}\n"
            f"{cyan}│{reset} SoftErrors: {self.softerror_count:<4}  "
            f"HardErrors: {self.harderror_count:<5} {cyan}│{reset}\n"
            f"{cyan}│{reset} {success_color}Success:{reset} {self.success*100:>6.1f}%  "
            f"Time: {self.time_s:>8.2f}s {cyan}│{reset}\n"
            f"{bottom}"
        )
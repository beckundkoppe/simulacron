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
    level_optimal_steps: int

    toolcall_count: float = 0
    observation_count: float = 0
    softerror_count: float = 0
    harderror_count: float = 0
    
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
            level_optimal_steps=base.level_optimal_steps,
            toolcall_count=mean("toolcall_count"),
            observation_count=mean("observation_count"),
            softerror_count=mean("softerror_count"),
            harderror_count=(mean("harderror_count")),
            success=(mean("success")),
            time_s=(mean("time_s")),
        )
    
    def toString(self, color: bool = True) -> str:
        """
        Formatiertes ASCII-Panel mit exakt ausgerichteten Spalten und dynamischer Breite.
        """
        import re

        # ANSI-Farben
        if color:
            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            CYAN = "\033[96m"
            RESET = "\033[0m"
        else:
            GREEN = RED = YELLOW = BLUE = CYAN = RESET = ""

        success_color = GREEN if self.success >= 0.8 else YELLOW if self.success >= 0.5 else RED

        # linke und rechte Spaltenbreite (anpassen, falls du längere Namen hast)
        left_width = 18
        right_width = 18

        # Hilfsfunktion, um ANSI-Codes beim Längenmessen zu ignorieren
        def strip_ansi(s: str) -> str:
            return re.sub(r'\x1b\[[0-9;]*m', '', s)

        def fmt_pair_fixed(left_label, left_val, right_label, right_val, right_start=20):
            left = f"{left_label}: {left_val}"
            right = f"{right_label}: {right_val}"
            # linke Spalte normal, rechte Spalte beginnt immer bei right_start
            spaces = right_start - len(strip_ansi(left))
            if spaces < 1:
                spaces = 1
            return f"{left}{' ' * spaces}{right}"

        lines = [
            f"{BLUE}Model:{RESET} {self.model_name}",
            f"{BLUE}Level:{RESET} {self.level_name}",
            f"{BLUE}Optimal Steps:{RESET} {self.level_optimal_steps}",
            #"-" * 40,
            fmt_pair_fixed("Toolcalls", f"{self.toolcall_count:.1f}", "Steps", f"{self.observation_count:.1f}"),
            fmt_pair_fixed("SoftErrors", f"{self.softerror_count:.1f}", "HardErrors", f"{int(self.harderror_count)}"),
            fmt_pair_fixed(f"{success_color}Success", f"{self.success*100:.1f}%{RESET}", "Time", f"{self.time_s:.2f}s")
        ]

        
        inner_width = max(len(strip_ansi(l)) for l in lines) + 2

        top_down_split_index = 3
        lines.insert(top_down_split_index, "-"*(inner_width-2))#

        top = f"{CYAN}┌{'─' * inner_width}┐{RESET}"
        bottom = f"{CYAN}└{'─' * inner_width}┘{RESET}"

        #def pad_line(s: str) -> str:
        #    return f"{CYAN}│{RESET} {s} {CYAN}│{RESET}"
        def pad_line(s: str) -> str:
            raw_len = len(strip_ansi(s))
            pad = inner_width - raw_len
            return f"{CYAN}│{RESET} {s}{' ' * (pad - 1)}{CYAN}│{RESET}"

        out = [top] + [pad_line(l) for l in lines] + [bottom]
        return "\n".join(out)
    


    
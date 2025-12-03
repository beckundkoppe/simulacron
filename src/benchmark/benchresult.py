from dataclasses import dataclass, fields
from enum import Enum

from llama_cpp import List

from benchmark.model_team import ModelTeam
from benchmark.run import Run
import json
from dataclasses import asdict, is_dataclass

from enviroment.levels.level import Level


@dataclass()
class PerformanceResult:
    run: Run
    hostname: str | None = None

    # 0.0 if failure, 1.0 if success
    success_rate: float = 0

    toolcall_count: float = 0

    # all agent toolcalls that interact with the enviroment (agents body entity)
    actions_external: float = 0

    # all toolcalls that are for e.g planning
    actions_internal: float = 0

    softerror_count: float = 0
    harderror_count: float = 0

    # PAL cycles
    step_count: float = 0

    # total time
    time_s: float = 0

    # time in agent state
    img_time_s: float = 0
    real_time_s: float = 0

    # time in agent state
    plan_time_s: float = 0 
    observe_time_s: float = 0 
    trial_time_s: float = 0
    action_time_s: float = 0
    reflect_time_s: float = 0

    @staticmethod
    def average(results: list["PerformanceResult"]) -> "PerformanceResult":
        n = len(results)
        if n == 0:
            raise ValueError("No results to average.")

        def mean(field: str) -> float:
            return sum(getattr(r, field) for r in results) / n

        base = results[0]
        return PerformanceResult(
            run=base.run,
            success_rate=mean("success_rate"),
            toolcall_count=mean("toolcall_count"),
            actions_external=mean("actions_external"),
            actions_internal=mean("actions_internal"),
            softerror_count=mean("softerror_count"),
            harderror_count=mean("harderror_count"),
            step_count=mean("step_count"),
            time_s=mean("time_s"),
            img_time_s=mean("img_time_s"),
            real_time_s=mean("real_time_s"),
            plan_time_s=mean("plan_time_s"),
            observe_time_s=mean("observe_time_s"),
            trial_time_s=mean("trial_time_s"),
            action_time_s=mean("action_time_s"),
            reflect_time_s=mean("reflect_time_s"),
        )

    def toJSON(self) -> str:
        """
        Gibt eine JSON-Zeichenkette zurück.
        Dataclasses (inkl. Run) werden korrekt in Dictionaries umgewandelt.
        """

        def serialize(obj):
            # 1. Wenn das Objekt eine toObject-Methode hat → zuerst verwenden
            if hasattr(obj, "toObject") and callable(obj.toObject):
                return serialize(obj.toObject())

            # 2. Enum behandeln
            if isinstance(obj, Enum):
                # Enthält der Enum ein Objekt mit toObject()? → Sonderfall
                if hasattr(obj.value, "toObject") and callable(obj.value.toObject):
                    return serialize(obj.value.toObject())
                return obj.name.lower()

            # 3. Dataclass → Felder manuell durchlaufen (aber nicht asdict!)
            if is_dataclass(obj):
                data = {}
                for f in fields(obj):
                    data[f.name] = serialize(getattr(obj, f.name))
                return data

            # 4. dict
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}

            # 5. list/tuple
            if isinstance(obj, (list, tuple)):
                return [serialize(x) for x in obj]

            # 6. primitive Werte
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            # 7. fallback
            return str(obj)
            

        data = serialize(self)
        return json.dumps(data, indent=4)



    def toString(self, color: bool = True) -> str:
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

        success_color = GREEN if self.success_rate >= 0.8 else YELLOW if self.success_rate >= 0.5 else RED

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

        def format_time(seconds: float) -> str:
            # Convert seconds to min:sec format if >= 60s
            if seconds >= 60:
                mins = int(seconds // 60)
                secs = seconds % 60
                return f"{mins}m {secs}s"
            return f"{seconds:.2f}s"

        model_name = self.run.model_team.label()
        host = self.hostname or "unknown"

        lines = [
            f"{BLUE}Model:{RESET} {model_name}",
            f"{BLUE}Level:{RESET} {self.run.level.value.getName()}",
            f"{BLUE}Optimal Steps:{RESET} {self.run.level.value.optimal_steps}",
            f"{BLUE}Host:{RESET} {host}",

            # Toolcalls and steps
            fmt_pair_fixed(
                "Toolcalls",
                f"{self.toolcall_count} ({self.actions_external}e/{self.actions_internal}i)",
                "Steps",
                f"{self.step_count:.1f}"
            ),

            # Soft and hard errors
            fmt_pair_fixed(
                "SoftErrors",
                f"{self.softerror_count:.1f}",
                "HardErrors",
                f"{int(self.harderror_count)}"
            ),

            # Success and time
            fmt_pair_fixed(
                f"{success_color}Success",
                f"{self.success_rate * 100:.1f}%{RESET}",
                "Time",
                format_time(self.time_s)  # here is the key change
            ),
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

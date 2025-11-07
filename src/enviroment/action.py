from abc import ABC
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

class ActionType(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    UNLOCK = "unlock"
    LOCK = "unlock"

@dataclass
class ActionResult:
    success: bool = ""
    result: str = ""
    error: str = ""

@dataclass
class ActionTry:
    type: ActionType 
    object_1: UUID = None
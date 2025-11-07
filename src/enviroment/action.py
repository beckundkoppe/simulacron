from abc import ABC
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

class ActionType(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    UNLOCK = "unlock"
    LOCK = "unlock"
    USE = "use"

@dataclass
class ActionTry:
    type: ActionType 
    item_1: UUID = None
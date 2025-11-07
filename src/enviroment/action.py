from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enviroment.entity import Entity

class ActionType(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    UNLOCK = "unlock"
    LOCK = "unlock"
    USE = "use"

@dataclass
class ActionTry:
    type: ActionType
    item_1: "Entity" | None = None

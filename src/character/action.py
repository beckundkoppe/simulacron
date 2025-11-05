from enum import Enum, auto

from advanced.tool import tool

class ActionType(Enum):
    TAKE    = "take",
    PUT     = "put",
    OPEN    = "open",
    CLOSE   = "close",
    USE     = "use",

#@tool
def perform_action(action: str, object: str) -> str:
    
    pass
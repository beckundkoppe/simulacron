import math
from typing import TYPE_CHECKING
from config import PositionType
import config

if TYPE_CHECKING:
    from enviroment.room import Room

class Position:
    def __init__(self, x: float = 0.0, y: float = 0.0, type: PositionType | None = None) -> None:
        self.x: float = x  # 0.0 = file a, 7.0 = file h
        self.y: float = y  # 0.0 = rank 8, 7.0 = rank 1
        self.type = type

    @staticmethod
    def from_input(x: str, y: str) -> "Position":
        return Position(float(x), float(y))
        
    def map(self, room) -> "Position":
        if not (self.type is None):
            raise ValueError("position cant only be mapped once")

        if config.ACTIVE_CONFIG.position == PositionType.ROOMLESS:
            return Position(0.0, 0.0, type=PositionType.ROOMLESS)
        else:
            return Position(self.x, self.y, type=PositionType.RELATIVE)
        
    def toString(self) -> str:
        if self.type is None:
            raise ValueError("only mapped positions can be transformed to string")

        if self.type == PositionType.ROOMLESS:
            return "in the current room"
        else:
            return f"({self.x},{self.y})"

    def distanceTo(self, pos: "Position") -> float:
        return math.sqrt((self.x - pos.x) ** 2 + (self.y - pos.y) ** 2)

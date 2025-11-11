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
    def from_input(x: str, y: str, room: "Room | None" = None) -> "Position":
        """Create a position from agent provided arguments.

        Depending on the active configuration the inputs are interpreted
        differently. For relative coordinates the arguments are parsed as
        floats. For the chessboard representation the inputs are treated as
        file and rank identifiers and transformed into the corresponding room
        coordinates.
        """

        active_config = getattr(config, "CONFIG", None)
        position_type = getattr(active_config, "position_type", PositionType.RELATIVE)

        if position_type == PositionType.CHESSBOARD:
            if room is None:
                raise ValueError("room must be provided for chessboard coordinates")

            field = f"{x}{y}"
            board_position = Position().fromChessboard(field)
            room_x = (board_position.x / 8.0) * room.extend_x
            room_y = (board_position.y / 8.0) * room.extend_y
            return Position(room_x, room_y)

        return Position(float(x), float(y))

    def fromChessboard(self, field: str) -> "Position":
        if len(field) != 2:
            raise ValueError("Chess field must be exactly 2 characters (e.g. 'a1')")

        file, rank = field[0].lower(), field[1]

        # File (column): a=0.0, b=1.0, ..., h=7.0
        if file < 'a' or file > 'h':
            raise ValueError("First character must be a-h")
        self.x = float(ord(file) - ord('a'))

        # Rank (row): 1 → y=7.0, 2 → y=6.0, ..., 8 → y=0.0
        if not rank.isdigit() or rank not in "12345678":
            raise ValueError("Second character must be 1-8")
        self.y = float(8 - int(rank))

        return self

    def toChessboard(self) -> str:
        if not (0 <= self.x <= 7 and 0 <= self.y <= 7):
            raise ValueError("position must be within chessboard (0-7)")
            
        file = chr(ord('a') + int(self.x))
        rank = str(8 - int(self.y))
        return f"{file}{rank}"
        
    def map(self, room) -> "Position":
        if not (self.type is None):
            raise ValueError("position cant only be mapped once")

        if config.ACTIVE_CONFIG.position == PositionType.ROOMLESS:
            return Position(0.0, 0.0, type=PositionType.ROOMLESS)
        elif config.ACTIVE_CONFIG.position == PositionType.CHESSBOARD:
            # Map room → 8x8 chessboard (0..7)
            chess_x = (self.x / room.extend_x) * 8.0
            chess_y = (self.y / room.extend_y) * 8.0
            return Position(chess_x, chess_y, type=PositionType.CHESSBOARD)   
        else:
            return Position(self.x, self.y, type=PositionType.RELATIVE)
        
    def toString(self) -> str:
        if self.type is None:
            raise ValueError("only mapped positions can be transformed to string")

        if self.type == PositionType.ROOMLESS:
            return "in the current room"
        elif self.type == PositionType.CHESSBOARD:
            return self.toChessboard()
        else:
            return f"({self.x},{self.y})"

    def distanceTo(self, pos: "Position") -> float:
        return math.sqrt((self.x - pos.x) ** 2 + (self.y - pos.y) ** 2)

from __future__ import annotations
from enum import Enum, auto
from typing import Set, Tuple
from uuid import UUID


from config import PositionType
from enviroment.interaction import Depth, Interaction, ObserverPerception, PerceptionEnviroment
from enviroment.world import World

class Room:
    def __init__(
        self,
        name: str | None,
        extend_x: float = 4.0,
        extend_y: float = 4.0,
        material: str | None = None,
        description: str | None = None,
        uniqueness: float = 0.5,
        light_level: float = 0.5,
        ambient_noise: float = 0.5,
        ambient_smell: float = 0.5
    ) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.postype = PositionType
        self.extend_x = float(extend_x)
        self.extend_y = float(extend_y)
        self.material = material
        self.description = description
        self.uniqueness = float(uniqueness)
        self.light = float(light_level)
        self.ambient_noise = float(ambient_noise)
        self.ambient_smell = float(ambient_smell)
        self.entities: Set[UUID] = set()

        self.uuid: UUID | None = None
        World.add_room(self)

    def perceive(self, observer: ObserverPerception, depth: Depth):
        o: str = ""
        for x in self.entities:
            entity = World.get_entity(x)

            env = PerceptionEnviroment(
                distance_m=0.5,
                ambient_smell=0.2,
                interactions=[Interaction.INSPECT, Interaction.FEEL]
            )

            o += str(entity.on_perceive(observer, env, depth)) + ", "

        return o


    def isPosInRoom(self, pos: Position) -> bool:
        if 0 <= pos.x < self.extend_x and 0 <= pos.y < self.extend_y:
            return True
        else:
            return False

    # ---------- factory methods ----------
    @classmethod
    def chamber(cls, name: str = "chamber") -> "Room":
        return cls(name, extend_x=1.0, extend_y=3.0)

    @classmethod
    def corridor(cls, length: float, name: str = "corridor") -> "Room":
        return cls(name, extend_x=length, extend_y=2.0)

class Position:
    def __init__(self, x: float = 0.0, y: float = 0.0, type: PositionType | None = None) -> None:
        self.x: float = x  # 0.0 = file a, 7.0 = file h
        self.y: float = y  # 0.0 = rank 8, 7.0 = rank 1
        self.type = type

    

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
        
    def map(self, room: Room) -> "Position":
        if not (self.type is None):
            raise ValueError("position cant only be mapped once")

        if room.postype == PositionType.ROOMLESS:
            return "Position"(0.0, 0.0, type=PositionType.ROOMLESS)  
        elif room.postype == PositionType.CHESSBOARD:
            # Map room → 8x8 chessboard (0..7)
            chess_x = (self.x / room.extend_x) * 8.0
            chess_y = (self.y / room.extend_y) * 8.0
            return "Position"(chess_x, chess_y, type=PositionType.CHESSBOARD)   
        else:
            return "Position"(self.x, self.y, type=PositionType.RELATIVE)
        
    def toString(self) -> str:
        if self.type is None:
            raise ValueError("only mapped positions can be transformed to string")
        
        if self.type == PositionType.ROOMLESS:
            return "in the current room"
        elif self.type == PositionType.CHESSBOARD:
            return self.toChessboard()
        else:
            return f"({self.x},{self.y})"
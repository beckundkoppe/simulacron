from __future__ import annotations
from typing import Set

from enviroment.resultbuffer import Resultbuffer


class World:
    entities: Set["Entity"] = set()
    rooms: Set["Room"] = set()
    _id_counter: int = 0

    @staticmethod
    def add_entity(entity: "Entity") -> None:
        if entity in World.entities:
            raise ValueError("Entity already belongs to a world")
        World._id_counter += 1
        entity.readable_id = f"{entity.name}_{World._id_counter}"

        World.entities.add(entity)

    @staticmethod
    def remove_entity(entity: "Entity") -> None:
        World.entities.discard(entity)

    @staticmethod
    def add_room(room: "Room") -> None:
        if room in World.rooms:
            raise ValueError("Room already belongs to a world")
        World._id_counter += 1
        room.readable_id = f"{room.name}_{World._id_counter}"

        World.rooms.add(room)

    @staticmethod
    def remove_room(room: "Room") -> None:
        World.rooms.discard(room)

    @staticmethod
    def clear():
        # Wipe execution results alongside world state so stale actions
        # from previous runs don't leak into the next level.
        Resultbuffer.buffer.clear()
        World.entities.clear()
        World.rooms.clear()
        World._id_counter = 0

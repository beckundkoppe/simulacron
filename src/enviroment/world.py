from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from enviroment.entity import Entity
    from enviroment.room import Room


class World:
    """Global registry that tracks all rooms and entities."""

    entities: Dict[int, "Entity"] = {}
    rooms: Dict[int, "Room"] = {}
    _id_counter: int = 0

    @staticmethod
    def _next_id() -> int:
        World._id_counter += 1
        return World._id_counter

    @staticmethod
    def add_entity(entity: "Entity") -> int:
        if entity.entity_id is not None:
            raise ValueError("Entity already belongs to a world")

        entity_id = World._next_id()
        entity.entity_id = entity_id
        entity.readable_id = f"{entity.name}_{entity_id}"

        World.entities[entity_id] = entity
        return entity_id

    @staticmethod
    def remove_entity(entity_id: int) -> None:
        World.entities.pop(entity_id, None)

    @staticmethod
    def get_entity(entity_id: int) -> Optional["Entity"]:
        return World.entities.get(entity_id)

    @staticmethod
    def add_room(room: "Room") -> int:
        if room.room_id is not None:
            raise ValueError("Room already belongs to a world")

        room_id = World._next_id()
        room.room_id = room_id
        room.readable_id = f"{room.name}_{room_id}"

        World.rooms[room_id] = room
        return room_id

    @staticmethod
    def remove_room(room_id: int) -> None:
        World.rooms.pop(room_id, None)

    @staticmethod
    def get_room(room_id: int) -> Optional["Room"]:
        return World.rooms.get(room_id)

    @staticmethod
    def clear() -> None:
        World.entities.clear()
        World.rooms.clear()
        World._id_counter = 0

from __future__ import annotations
from typing import Dict, Optional

import uuid
from uuid import UUID


class World:
    entities: Dict[UUID, "Entity"] = {}
    rooms: Dict[UUID, "Room"] = {}
    _id_counter: int = 0

    @staticmethod
    def add_entity(entity: "Entity") -> UUID:
        if entity.uuid is not None:
            raise ValueError("Entity already belongs to a world")

        World._id_counter += 1
        new_uuid = uuid.uuid4()
        entity.uuid = new_uuid
        entity.readable_id = f"{entity.name}_{World._id_counter}"

        World.entities[new_uuid] = entity
        return new_uuid

    @staticmethod
    def remove_entity(uuid: UUID) -> None:
        entity = World.entities.pop(uuid, None)
        if entity:
            entity.world = None

    @staticmethod
    def get_entity(uuid: UUID) -> Optional["Entity"]:
        return World.entities.get(uuid)

    @staticmethod
    def add_room(room: "Room") -> UUID:
        World._id_counter += 1
        new_uuid = uuid.uuid4()
        room.uuid = new_uuid
        room.readable_id = f"{room.name}_{World._id_counter}"

        World.rooms[new_uuid] = room
        return new_uuid

    @staticmethod
    def remove_room(uuid: UUID) -> None:
        room = World.rooms.pop(uuid, None)
        if room:
            room.world = None

    @staticmethod
    def get_room(uuid: UUID) -> Optional["Room"]:
        return World.rooms.get(uuid)
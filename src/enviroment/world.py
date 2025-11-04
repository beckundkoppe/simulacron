from __future__ import annotations
from typing import Dict

import uuid
from uuid import UUID

from benchmark.level import Level


class World:
    def __init__(self) -> None:
        self.entities: Dict[UUID, Entity] = {}
        self.rooms: Dict[UUID, Room] = {}
        self._id_counter = 0

    def add_entity(self, entity: Entity) -> UUID:
        if entity.uuid is not None:
            raise ValueError("Entity already belongs to a world")

        self._id_counter += 1
        new_uuid = uuid.uuid4()
        entity.uuid = new_uuid
        entity.readable_id = f"{entity.name}_{self._id_counter}"
        entity.world = self

        self.entities[new_uuid] = entity
        return new_uuid
    
    def remove_entity(self, uuid: UUID) -> None:
        entity = self.entities.pop(uuid, None)
        if entity:
            entity.world = None

    def get_entity(self, uuid: UUID) -> Entity | None:
        return self.entities.get(uuid)

    def add_room(self, room: Room) -> UUID:
        if entity.uuid is not None:
            raise ValueError("Entity already belongs to a world")

        self._id_counter += 1
        new_uuid = uuid.uuid4()
        room.uuid = new_uuid
        room.readable_id = f"{room.name}_{self._id_counter}"
        room.world = self

        self.rooms[new_uuid] = room
        return new_uuid

    def remove_room(self, uuid: UUID) -> None:
        room = self.rooms.pop(uuid, None)
        if room:
            room.world = None

    def get_room(self, uuid: UUID) -> Room:
        return self.entities.get(uuid)

        
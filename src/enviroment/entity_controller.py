

from __future__ import annotations
from uuid import UUID
from enviroment.world import World


class EntityController:
    def __init__(self, entity):
        self.entity: UUID = entity.uuid

    def move_to_object(self, object: UUID):
        en = World.get_entity(self.entity)
        ob = World.get_entity(object)

        en.pos = ob.pos
        
    def move_to_position(self, pos):
        en = World.get_entity(self.entity)

        en.pos = pos
        
    def use_connector(self, connector_uuid: UUID):
        entity = World.get_entity(self.entity)
        connector = World.get_entity(connector_uuid)
        connector.enter_connect(entity)



   # def pick_up(self, entity_uuid: UUID):
        

        
        
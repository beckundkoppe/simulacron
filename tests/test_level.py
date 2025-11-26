import sys
from pathlib import Path
import unittest

from enviroment.room import Room

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from enviroment.action import ActionTry, ActionType
from enviroment.levels.data import LevelSpec
from enviroment.levels import carrot, cucumber, onion, potato, salad, tomato
from enviroment.world import World
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, Entity, ContainerEntity



class TestLevel(unittest.TestCase):
    def _extract_entity(self, name):
        for e in World.entities:
            if e.name == name:
                return e
        return None
        
    def _get_from_inventory(self, name, agent):
        for e in agent.inventory:
            if e.name == name:
                return e
        return None
    
    def _extract_room(self, name):
        for e in World.rooms:
            if e.name == name:
                return e
        return None
    
    def _find_connector(self, room_a: Room, room_b: Room) -> ConnectorEntity:
        for ent in room_a.entities:
            if isinstance(ent, ConnectorEntity) and room_b in ent.rooms:
                return ent
        raise AssertionError(f"No connector between {room_a.name} and {room_b.name}")


    def setUp(self) -> None:
        World.clear()

    def tearDown(self) -> None:
        World.clear()

    def test_level_carrot_easy(self):
        lvl: LevelSpec = carrot.build_easy(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        target = self._extract_entity("carrot")
        self.assertIsNotNone(target)

        self.assertIn(target, agent.inventory)
    
        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        self.assertFalse(lvl.is_success())

        agent.move_to_object(table)
        agent.drop_into(target, table)

        self.assertTrue(lvl.is_success())
    def test_level_carrot_hard(self):
        lvl: LevelSpec = carrot.build_hard(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        target = self._extract_entity("carrot")
        self.assertIsNotNone(target)

        self.assertIn(target, agent.inventory)
        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        self.assertFalse(lvl.is_success())

        agent.move_to_object(table)
        agent.drop_into(target, table)

        self.assertTrue(lvl.is_success())

    def test_level_cucumber_easy(self):
        lvl: LevelSpec = cucumber.build_easy(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        cuc = self._extract_entity("cucumber")
        self.assertIsNotNone(cuc)

        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        self.assertFalse(lvl.is_success())

        agent.move_to_object(table)
        agent.take(cuc)
        agent.drop_into(cuc, table)

        self.assertTrue(lvl.is_success())
    def test_level_cucumber_hard(self):
        lvl: LevelSpec = cucumber.build_hard(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        cuc = self._extract_entity("cucumber")
        self.assertIsNotNone(cuc)

        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        self.assertFalse(lvl.is_success())

        agent.move_to_object(table)
        agent.take(cuc)
        agent.drop_into(cuc, table)

        self.assertTrue(lvl.is_success())

    def test_level_salad_easy(self):
        lvl: LevelSpec = salad.build_easy(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Entities extrahieren
        box = self._extract_entity("box")
        self.assertIsNotNone(box)
        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)

        # Listen der Items
        spinach_items = [e for e in World.entities if e.name == "spinach"]
        lettuce_items = [e for e in World.entities if e.name == "lettuce"]

        self.assertEqual(len(spinach_items), 3)
        self.assertEqual(len(lettuce_items), 3)

        # Zu Beginn darf der Level nicht erfolgreich sein
        self.assertFalse(lvl.is_success())

        # ----------- Spinach korrekt einräumen -----------
        for item in spinach_items:
            agent.move_to_object(item)
            agent.take(item)
            agent.move_to_object(box)
            agent.drop_into(item, box)

        # ----------- Lettuce korrekt einräumen -----------
        for item in lettuce_items:
            agent.move_to_object(item)
            agent.take(item)
            agent.move_to_object(shelf)
            agent.drop_into(item, shelf)

        # Jetzt sollte das Level gewonnen sein
        self.assertTrue(lvl.is_success())
    def test_level_salad_hard(self):
        lvl: LevelSpec = salad.build_heard(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Entities extrahieren
        box = self._extract_entity("box")
        self.assertIsNotNone(box)
        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)

        # Listen der Items
        spinach_items = [e for e in World.entities if e.name == "spinach"]
        lettuce_items = [e for e in World.entities if e.name == "lettuce"]

        self.assertEqual(len(spinach_items), 3)
        self.assertEqual(len(lettuce_items), 3)

        # Zu Beginn darf der Level nicht erfolgreich sein
        self.assertFalse(lvl.is_success())

        # ----------- Spinach korrekt einräumen -----------
        for item in spinach_items:
            agent.move_to_object(item)
            agent.take(item)
            agent.move_to_object(box)
            agent.drop_into(item, box)

        # ----------- Lettuce korrekt einräumen -----------
        for item in lettuce_items:
            agent.move_to_object(item)
            agent.take(item)
            agent.move_to_object(shelf)
            agent.drop_into(item, shelf)

        # Jetzt sollte das Level gewonnen sein
        self.assertTrue(lvl.is_success())

    def test_level_tomato_easy(self):
        lvl: LevelSpec = tomato.build_easy(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Objekte extrahieren
        tomato_ent = self._extract_entity("tomato")
        self.assertIsNotNone(tomato_ent)

        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        box: ContainerEntity = self._extract_entity("box")
        self.assertIsNotNone(box)

        crate = self._extract_entity("chest")
        self.assertIsNotNone(crate)

        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)

        self.assertFalse(lvl.is_success())

        # 1. Tomate vom Tisch nehmen
        agent.move_to_object(table)
        # 2. Tomate nehmen
        agent.take_from(tomato_ent, table)
        # 3. Zur Box gehen
        agent.move_to_object(box)
        # 4. die Tomate in die Box legen
        agent.drop_into(tomato_ent, box)
        # 5. Box schließen
        box.on_interact(agent, ActionTry(ActionType.CLOSE))
        # 6. Box nehmen
        agent.take(box)
        # 7 zur Kiste gehen
        agent.move_to_object(crate)
        # 8. Box in die Kiste legen
        agent.drop_into(box, crate)
        # 9. Kiste schließen
        crate.on_interact(agent, ActionTry(ActionType.CLOSE))
        # 10. Kiste nehmen
        agent.take(crate)
        # 11. zum Regal gehen
        agent.move_to_object(shelf)
        # 12. Kiste in Regal stellen
        agent.drop_into(crate, shelf)

        self.assertTrue(lvl.is_success())
    def test_level_tomato_hard(self):
        lvl: LevelSpec = tomato.build_hard(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Objekte extrahieren
        tomato_ent = self._extract_entity("tomato")
        self.assertIsNotNone(tomato_ent)

        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        box: ContainerEntity = self._extract_entity("box")
        self.assertIsNotNone(box)

        crate = self._extract_entity("chest")
        self.assertIsNotNone(crate)

        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)

        self.assertFalse(lvl.is_success())

        # 1. Tomate vom Tisch nehmen
        agent.move_to_object(table)
        # 2. Tomate nehmen
        agent.take_from(tomato_ent, table)
        # 3. Zur Box gehen
        agent.move_to_object(box)
        # 4. die Tomate in die Box legen
        agent.drop_into(tomato_ent, box)
        # 5. Box schließen
        box.on_interact(agent, ActionTry(ActionType.CLOSE))
        # 6. Box nehmen
        agent.take(box)
        # 7 zur Kiste gehen
        agent.move_to_object(crate)
        # 8. Box in die Kiste legen
        agent.drop_into(box, crate)
        # 9. Kiste schließen
        crate.on_interact(agent, ActionTry(ActionType.CLOSE))
        # 10. Kiste nehmen
        agent.take(crate)
        # 11. zum Regal gehen
        agent.move_to_object(shelf)
        # 12. Kiste in Regal stellen
        agent.drop_into(crate, shelf)

        self.assertTrue(lvl.is_success())

    def test_level_potato_easy(self):
        lvl: LevelSpec = potato.build_easy(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Entities extrahieren
        table = self._extract_entity("table")
        self.assertIsNotNone(table)
        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)
        potato_ent = self._extract_entity("potato")
        self.assertIsNotNone(potato_ent)
        chest = self._extract_entity("chest")
        self.assertIsNotNone(chest)
        door = self._extract_entity("door")
        self.assertIsNotNone(door)

        main_room = self._extract_room("main")
        self.assertIsNotNone(main_room)
        chamber_room = self._extract_room("chamber")
        self.assertIsNotNone(chamber_room)
        self.assertTrue(main_room.contains_entity(agent))
        self.assertTrue(main_room.contains_entity(table))
        self.assertTrue(chamber_room.contains_entity(shelf))

        self.assertFalse(lvl.is_success())

        # 1. Zur Tür gehen
        agent.move_to_object(door)

        # 2. Durch die Tür in die Kammer gehen
        agent.use_connector(door)

        # 3. Zum shelf gehen
        agent.move_to_object(shelf)
        
        # 4. Kartoffel aus Kiste nehmen
        agent.take_from(potato_ent, chest)

        # 5. zurück zur Tür gehen
        agent.move_to_object(door)
        
        # 6. in main zurückgehen
        agent.use_connector(door)

        # 7. Zum tisch gehen
        agent.move_to_object(table)

        # 8. Potato auf den Tisch legen
        agent.drop_into(potato_ent, table)

        self.assertTrue(lvl.is_success())
    def test_level_potato_medium(self):
        lvl: LevelSpec = potato.build_medium(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Entities extrahieren
        table = self._extract_entity("table")
        self.assertIsNotNone(table)
        shelf = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)
        potato_ent = self._extract_entity("potato")
        self.assertIsNotNone(potato_ent)
        chest = self._extract_entity("chest")
        self.assertIsNotNone(chest)
        door = self._extract_entity("door")
        self.assertIsNotNone(door)

        main_room = self._extract_room("main")
        self.assertIsNotNone(main_room)
        chamber_room = self._extract_room("chamber")
        self.assertIsNotNone(chamber_room)
        self.assertTrue(main_room.contains_entity(agent))
        self.assertTrue(main_room.contains_entity(table))
        self.assertTrue(chamber_room.contains_entity(shelf))

        self.assertFalse(lvl.is_success())

        # 1. Zur Tür gehen
        agent.move_to_object(door)

        # 2. Durch die Tür in die Kammer gehen
        agent.use_connector(door)

        # 3. Zum shelf gehen
        agent.move_to_object(shelf)
        
        # 4. Kartoffel aus Kiste nehmen
        agent.take_from(potato_ent, chest)

        # 5. zurück zur Tür gehen
        agent.move_to_object(door)
        
        # 6. in main zurückgehen
        agent.use_connector(door)

        # 7. Zum tisch gehen
        agent.move_to_object(table)

        # 8. Potato auf den Tisch legen
        agent.drop_into(potato_ent, table)

        self.assertTrue(lvl.is_success())
    def test_level_potato_hard(self):
        lvl: LevelSpec = potato.build_hard(True)

        agent: AgentEntity
        instruction: str
        agent, instruction = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        # Entities extrahieren
        table = self._extract_entity("table")
        self.assertIsNotNone(table)
        shelf: AdvancedContainerEntity = self._extract_entity("shelf")
        self.assertIsNotNone(shelf)
        potato_ent = self._extract_entity("potato")
        self.assertIsNotNone(potato_ent)
        chest = self._extract_entity("chest")
        self.assertIsNotNone(chest)
        door = self._extract_entity("door")
        self.assertIsNotNone(door)
        key = self._extract_entity("key")
        self.assertIsNotNone(key)

        main_room = self._extract_room("main")
        self.assertIsNotNone(main_room)
        chamber_room = self._extract_room("chamber")
        self.assertIsNotNone(chamber_room)
        self.assertTrue(main_room.contains_entity(agent))
        self.assertTrue(main_room.contains_entity(table))
        self.assertTrue(chamber_room.contains_entity(shelf))

        self.assertFalse(lvl.is_success())

        # 1. Zum Tisch gehen
        agent.move_to_object(table)
        # 2. Schlüssel vom Tisch nehmen
        agent.take_from(key, table)
        # 3. Zur Tür gehen
        agent.move_to_object(door)
        # 4. Durch die Tür in die Kammer gehen
        agent.use_connector(door)
        # 5. Zum Regal gehen
        agent.move_to_object(shelf)
        # 6. Kiste mit dem Schlüssel aufschließen
        chest.on_interact(agent, ActionTry(ActionType.UNLOCK, key))
        # 7. Kiste öffnen
        chest.on_interact(agent, ActionTry(ActionType.OPEN))
        # 8. Kartoffel aus der Kiste nehmen
        agent.take_from(potato_ent, chest)
        # 9. Zur Tür zurückgehen
        agent.move_to_object(door)
        # 10. In den Hauptraum zurückgehen
        agent.use_connector(door)
        # 11. Zum Tisch gehen
        agent.move_to_object(table)
        # 12. Kartoffel auf den Tisch legen
        agent.drop_into(potato_ent, table)

        self.assertTrue(lvl.is_success())

    def test_onion_easy(self):
        lvl = onion.build_easy(detailed_instruction=False)
        agent: AgentEntity
        agent, _ = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        target = self._extract_entity("onion")
        self.assertIsNotNone(target)
        compost = self._extract_entity("compost")
        self.assertIsNotNone(compost)
        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        living_room = self._extract_room("living_room")
        self.assertIsNotNone(living_room)
        self.assertTrue(living_room.contains_entity(agent))
        self.assertTrue(living_room.contains_entity(table))
        self.assertTrue(table.contains_entity(target))
        
        garden = self._extract_room("garden")
        self.assertIsNotNone(garden)
        self.assertTrue(garden.contains_entity(compost))

        hallway = self._extract_room("hallway")
        self.assertIsNotNone(hallway)
        mudroom = self._extract_room("mudroom")
        self.assertIsNotNone(mudroom)

        
        door_living_hallway = self._find_connector(living_room, hallway)
        self.assertIsNotNone(door_living_hallway)
        door_hallway_mudroom = self._find_connector(hallway, mudroom)
        self.assertIsNotNone(door_hallway_mudroom)
        door_mudroom_garden = self._find_connector(mudroom, garden)
        self.assertIsNotNone(door_mudroom_garden)

        self.assertFalse(lvl.is_success())

        # 1. Zum Tisch gehen
        agent.move_to_object(table)

        # 2. Zwiebel vom Tisch nehmen
        agent.take_from(target, table)

        # 3. Zur Tür in den Flur gehen
        agent.move_to_object(door_living_hallway)

        # 4. In den Flur wechseln
        agent.use_connector(door_living_hallway)

        # 5. Zur Tür in den Vorraum (mudroom) gehen
        agent.move_to_object(door_hallway_mudroom)

        # 6. In den Vorraum wechseln
        agent.use_connector(door_hallway_mudroom)

        # 7. Zur Tür in den Garten gehen
        agent.move_to_object(door_mudroom_garden)

        # 8. In den Garten wechseln
        agent.use_connector(door_mudroom_garden)

        # 9. Zum Kompost gehen
        agent.move_to_object(compost)

        # 10. Zwiebel in den Kompost legen
        agent.drop_into(target, compost)

        self.assertTrue(lvl.is_success())

    def test_onion_hard(self):
        lvl = onion.build_hard(detailed_instruction=False)
        agent: AgentEntity
        agent, _ = lvl.agent_entities[0]
        self.assertIsNotNone(agent)

        target = self._extract_entity("onion")
        self.assertIsNotNone(target)
        compost = self._extract_entity("compost")
        self.assertIsNotNone(compost)
        table = self._extract_entity("table")
        self.assertIsNotNone(table)

        living_room = self._extract_room("living_room")
        self.assertIsNotNone(living_room)
        self.assertTrue(living_room.contains_entity(agent))
        self.assertTrue(living_room.contains_entity(table))
        self.assertTrue(table.contains_entity(target))
        
        garden = self._extract_room("garden")
        self.assertIsNotNone(garden)
        self.assertTrue(garden.contains_entity(compost))

        hallway = self._extract_room("hallway")
        self.assertIsNotNone(hallway)
        mudroom = self._extract_room("mudroom")
        self.assertIsNotNone(mudroom)

        
        door_living_hallway = self._find_connector(living_room, hallway)
        self.assertIsNotNone(door_living_hallway)
        door_hallway_mudroom = self._find_connector(hallway, mudroom)
        self.assertIsNotNone(door_hallway_mudroom)
        door_mudroom_garden = self._find_connector(mudroom, garden)
        self.assertIsNotNone(door_mudroom_garden)

        self.assertFalse(lvl.is_success())

        # 1. Zum Tisch gehen
        agent.move_to_object(table)

        # 2. Zwiebel vom Tisch nehmen
        agent.take_from(target, table)

        # 3. Zur Tür in den Flur gehen
        agent.move_to_object(door_living_hallway)

        # 4. In den Flur wechseln
        agent.use_connector(door_living_hallway)

        # 5. Zur Tür in den Vorraum (mudroom) gehen
        agent.move_to_object(door_hallway_mudroom)

        # 6. In den Vorraum wechseln
        agent.use_connector(door_hallway_mudroom)

        # 7. Zur Tür in den Garten gehen
        agent.move_to_object(door_mudroom_garden)

        # 8. In den Garten wechseln
        agent.use_connector(door_mudroom_garden)

        # 9. Zum Kompost gehen
        agent.move_to_object(compost)

        # 10. Zwiebel in den Kompost legen
        agent.drop_into(target, compost)

        self.assertTrue(lvl.is_success())

if __name__ == "__main__":
    unittest.main()

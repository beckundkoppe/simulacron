"""Regressions for the in-game entities and interaction rules.

This module exercises movement, inventory handling, container/connector
behaviour and interaction radius constraints so the environment can be
smoke-tested quickly.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from enviroment.action import ActionTry, ActionType
from enviroment.entity import (
    AdvancedContainerEntity,
    AgentEntity,
    ConnectorEntity,
    ContainerEntity,
    Entity,
)
from enviroment.exception import HardException, SoftException
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World


class TestEnvInteraction(unittest.TestCase):
    def setUp(self) -> None:
        World.clear()
        self.room = Room.default()
        self.agent = AgentEntity("agent", Position(0.5, 0.5))
        self.agent.enter(self.room)

    def tearDown(self) -> None:
        World.clear()

    def test_move_actions_require_valid_room_and_bounds(self) -> None:
        """Agent must stay within room bounds and can move to visible objects."""

        target = Position(1.25, 1.25)
        self.agent.move_to_position(target)
        self.assertEqual((self.agent.pos.x, self.agent.pos.y), (target.x, target.y))

        other = Entity("crate", Position(0.25, 0.25))
        other.enter(self.room)
        self.agent.move_to_object(other)
        self.assertEqual((self.agent.pos.x, self.agent.pos.y), (other.pos.x, other.pos.y))

        unreachable_room = Room.default(name="side-room")
        distant_object = Entity("far", Position(0.5, 0.5))
        distant_object.enter(unreachable_room)
        with self.assertRaises(HardException):
            self.agent.move_to_object(distant_object)

        with self.assertRaises(SoftException):
            self.agent.move_to_position(Position(self.room.extend_x + 1.0, self.room.extend_y + 1.0))

    def test_take_and_drop_on_floor_respects_inventory_rules(self) -> None:
        """Taking and dropping on the floor manipulates inventory consistently."""

        item = Entity("item", Position(0.5, 0.4))
        item.enter(self.room)
        self.agent.take(item)
        self.assertIn(item, self.agent.inventory)

        with self.assertRaises(HardException):
            self.agent.drop(Entity("ghost", Position(0, 0)))

        self.agent.drop(item)
        self.assertNotIn(item, self.agent.inventory)
        self.assertTrue(item.is_collectible)
        self.assertEqual(item.room, self.room)

    def test_take_from_and_drop_into_container(self) -> None:
        """Container children can be removed and reinserted by the agent."""

        container = ContainerEntity("box", Position(1.0, 0.5))
        container.enter(self.room)

        stored = Entity("stored", Position(1.0, 0.5))
        container.add_child(stored)

        self.agent.take_from(stored, container)
        self.assertIn(stored, self.agent.inventory)
        self.assertFalse(container.contains_entity(stored))

        self.agent.drop_into(stored, container)
        self.assertFalse(stored in self.agent.inventory)
        self.assertTrue(container.contains_entity(stored))

        with self.assertRaises(HardException):
            self.agent.take_from(Entity("missing", Position(0, 0)), container)

    def test_advanced_container_locking_and_interaction_radius(self) -> None:
        """Lock/unlock/open flows respect range limits and required keys."""

        key = Entity("key", Position(0.5, 0.5))
        key.enter(self.room)
        chest = AdvancedContainerEntity(
            "chest",
            Position(1.0, 0.5),
            is_open=False,
            is_locked=True,
        )
        chest.enter(self.room)
        chest.register_key(key)

        with self.assertRaises(SoftException):
            chest.on_interact(self.agent, ActionTry(ActionType.OPEN))

        self.agent.take(key)
        chest.on_interact(self.agent, ActionTry(ActionType.UNLOCK, key))
        self.assertFalse(chest.is_locked)
        self.assertIn("opened", chest.on_interact(self.agent, ActionTry(ActionType.OPEN)))

        self.agent.move_to_position(Position(3.5, 3.5))
        with self.assertRaises(SoftException):
            chest.on_interact(self.agent, ActionTry(ActionType.CLOSE))

    def test_connector_usage_and_locking_rules(self) -> None:
        """Connectors enforce the allowed operators and lock state."""

        second_room = Room.default(name="hall")
        entrance = ConnectorEntity("door_a", Position(1.0, 0.5))
        exit = ConnectorEntity("door_b", Position(1.0, 0.5))
        entrance.connect(exit)
        exit.connect(entrance)
        entrance.enter(self.room)
        exit.enter(second_room)

        key = Entity("key", Position(0.5, 0.5))
        key.enter(self.room)
        entrance.add_key(key)
        exit.add_key(key)

        self.assertIn("Went through", entrance.on_interact(self.agent, ActionTry(ActionType.USE)))
        self.assertEqual(self.agent.room, second_room)
        self.assertIn("Went through", exit.on_interact(self.agent, ActionTry(ActionType.USE)))
        self.assertEqual(self.agent.room, self.room)

        entrance.is_locked = True

        wrong_key = Entity("wrong", Position(1.0, 0.5))
        wrong_key.enter(self.room)
        with self.assertRaises(SoftException):
            entrance.on_interact(self.agent, ActionTry(ActionType.UNLOCK, wrong_key))

        with self.assertRaises(SoftException):
            entrance.on_interact(self.agent, ActionTry(ActionType.USE))

        self.agent.take(key)
        entrance.on_interact(self.agent, ActionTry(ActionType.UNLOCK, key))
        self.assertFalse(entrance.is_locked)
        self.assertIn("Went through", entrance.on_interact(self.agent, ActionTry(ActionType.USE)))


if __name__ == "__main__":
    unittest.main()
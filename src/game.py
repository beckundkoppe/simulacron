import json
from agent.agent import Agent
import config
from config import PositionType
import current
from enviroment.entity import AgentEntity
from enviroment.levels.level import Level, LevelSpec
from util import console
from enviroment.perception import DetailLevel
from enviroment.position import Position
from enviroment.room import Room


def _position_payload(pos: Position, room: Room) -> tuple[str, object]:
    if getattr(config, "CONFIG", None) is None:
        return ("relative", {"x": pos.x, "y": pos.y})

    mapped = pos.map(room)
    if mapped.type == PositionType.CHESSBOARD:
        return ("chessboard", mapped.toString())
    if mapped.type == PositionType.ROOMLESS:
        return ("roomless", mapped.toString())
    return ("relative", {"x": mapped.x, "y": mapped.y})


def observe(observer: AgentEntity) -> str:
    room = observer.room

    position_format, position_value = _position_payload(observer.pos, room)
    data = {}
    data["you_are_in_room"] = {
        "name": room.name,
        "your_pos": position_value,
        "room_size": {
            "extend_x": room.extend_x,
            "extend_y": room.extend_y,
        }
    }
    data["your_inventory"] = observer.get_inventory()
    data["your_observation"] = room.perceive(observer, DetailLevel.OMNISCIENT)
    return json.dumps(data)

def run_level(level: Level, optimal_steps_multilier: float, main_model, imaginator = None, extra_model = None):
    if(imaginator is None):
        imaginator = main_model
    if(extra_model is None):
        extra_model = main_model

    current.EXTRA_MODEL = extra_model

    spec: LevelSpec = level.build()
    console.pretty(console.banner(level.name, char="+", color=console.Color.BLUE))

    agents = []
    success = False

    try:
        for entity, prompt in spec.agent_entities:
            console.pretty(console.bullet(f"{entity.name}\t[PROMPT:] {prompt}", color=console.Color.BLUE))

            agent = Agent(goal=prompt, entity=entity, imaginator_model=imaginator, realisator_model=main_model)
            agents.append(agent)


        for i in range(int(level.optimal_steps * optimal_steps_multilier)):
            console.pretty(console.bullet(f"Observation: {i}", color=console.Color.BLUE))

            for agent in agents:
                current.RESULT.observation_count += 1

                observation = observe(agent.entity)
                agent.update(observation)

                if spec.is_success():
                    success = True
                    break

            if success:
                console.pretty(console.banner(f"Finished after {i} steps", char="+", color=console.Color.BLUE))
                break

        current.RESULT.success = 1 if success else 0

    #except Exception as a:
        #console.pretty(console.banner(f"Execution failed: {str(a)}", color=console.Color.RED))
    finally:
        pass

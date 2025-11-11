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

#class SingleAgentTeam(AgentTeam):
#    agent: ToolProvider
#
#    def __init__(self, task: str, entity: AgentEntity, runner: Provider, extra: Provider):
#        agent_mem = SummarizingMemory(model=extra)
#        agent_mem.add_message(Role.SYSTEM, task)
#
#        self.agent = ToolProvider.build(runner, entity=entity, memory=agent_mem)
#
#    def get_entity(self) -> Entity:
#        return self.agent.entity
#
#    def step(self, observations: str) -> str:
#        current.AGENT = self.agent
#        print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observations))!s}", color=console.Color.CYAN))
#
#        _update_move_tool_description()
#        self.agent.register_tools(TOOLS)
#        self.agent.memory.save("mem.txt")
#        self.agent.invoke(observations, "Give best next action.")
#
#        process_results(self.agent)
#        current.AGENT = None
#
#class TwoAgentTeam(AgentTeam):
#    imaginator: ToolProvider
#    realisator: ToolProvider
#
#    def __init__(self, task: str, AgentEntity, imaginator: Provider, realisator: Provider, extra: Provider):
#        self.imaginator = imaginator
#        self.realisator = realisator
#        img_mem = SummarizingMemory(model=extra)
#        img_mem.add_message(Role.SYSTEM, task)
#
#        self.imaginator = ToolProvider.build(imaginator, entity=entity, memory=img_mem, name="imaginator")
#
#        real_mem = self._build_realisator_memory()
#
#        self.realisator = ToolProvider.build(realisator, entity=entity, memory=real_mem, name="realisator")
#        self._known_suggestions: Set[str] = set()
#
#    def get_entity(self) -> Entity:
#        return self.imaginator.entity
#
#    def step(self, observations: str) -> str:
#        current.AGENT = self.imaginator
#        print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observations))!s}", color=console.Color.CYAN))
#        self.imaginator.register_tools(None)
#
#        self.imaginator.memory.save("mem.txt")
#        imagination = self.imaginator.invoke(observations, "Give best next action (short)")
#
#        _update_move_tool_description()
#        self._reset_realisator_memory()
#
#        keep_alive = True
#        while keep_alive:
#            self.realisator.register_tools(TOOLS)
#            #process_formal_errors(None) #delete formal errors from unallowed
#            reply = self.realisator.invoke(observations + " PLAN: " + imagination, "Give the toolcalls that arise from the plan. (if something is unclear answer with a precice and short question). ")
#            keep_alive = process_formal_errors(self.realisator)
#            #print(">" + reply + "<")
#
#            if(len(reply) > 0):
#                #self.imaginator.memory.add_message(Role.USER, reply)
#                self.imaginator.memory.add_message(Role.USER, "last action was not specified well. please provide more explicit instruction")
#                keep_alive = False
#
#        process_action_results(self.imaginator)
#
#        current.AGENT = None
#
#    def _build_realisator_memory(self, *, learning_mode: bool = False) -> Memory:
#        mem = Memory()
#        mem.add_message(
#            Role.SYSTEM,
#            (
#                "Execute the plans with precise tool calls, "
#                "and keep your replies minimal unless you must ask for clarification."
#            ),
#        )
#
#        return mem
#
#    def _reset_realisator_memory(self, *, learning_mode: bool = False) -> None:
#        self.realisator.memory = self._build_realisator_memory(learning_mode=learning_mode)

def run_level(level: Level, optimal_steps_multilier: float, cache, main_model, realisator = None, extra_model = None):
    if(realisator is None):
        realisator = main_model
    if(extra_model is None):
        extra_model = main_model

    spec: LevelSpec = level.build()
    console.pretty(console.banner(level.name, char="+", color=console.Color.BLUE))

    agents = []
    success = False

    try:
        for entity, prompt in spec.agent_entities:
            console.pretty(console.bullet(f"{entity.name}\t[PROMPT:] {prompt}", color=console.Color.BLUE))

            agent = Agent(cache, main_model, realisator, extra_model)
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

    except Exception as a:
        console.pretty(console.banner(f"Execution failed: {str(a)}", color=console.Color.RED))

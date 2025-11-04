# test_main.py

from advanced.agent import Agent
from benchmark import benchmark
import debug.settings as settings
from llm.cache import Cache
from advanced.tool import tool
from llm.memory.memory import Memory
from llm.model import Model
from llm.runner import Role, Runner

@tool
def get_time() -> str:
    """Return the current system time as an ISO string.

    Returns:
        str: Current system time in ISO 8601 format.
    """
    
    settings.pretty(
        settings.bullet(f"[TOOLCALL] get_time", color=settings.Color.YELLOW),
    )
    return "19:20"


@tool
def skip() -> None:
    """End the agent's turn.

    This tool signals that no further actions are required.

    Returns:
        None
    """
    settings.pretty(
        settings.bullet(f"[TOOLCALL] skip", color=settings.Color.YELLOW),
    )

@tool
def body_say(msg: str) -> str:
    """The human says something something loud.

    Args:
        msg (str): The message text to output.

    Returns:
        str: A confirmation string echoing the supplied message.
    """
    settings.pretty(
        settings.bullet(f"[TOOLCALL] say: {msg}", color=settings.Color.YELLOW),
    )
    return f"You said: {msg}"

@tool
def perform_body_action(action: str) -> str:
    """Perform a physical action with the human body.

    Args:
        action (str): The description of the ction to perform.

    Returns:
        str: A confirmation string.
    """
    settings.pretty(
        settings.bullet(f"[TOOLCALL] perform_body_action: {action}", color=settings.Color.YELLOW),
    )
    return f"You did: {action}"

def main():
    benchmark.full_run()

    cache = Cache()

    cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)

    ##### Runner Example

    runner_memory = Memory()
    runner = cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    runner.invoke("hello, im felix", role=Role.USER, memory=runner_memory)
    runner.invoke("What did i say my name was?", role=Role.USER, memory=runner_memory)

    ##### Agent Example

    agent_memory = Memory()
    agent_memory.add_message(Role.SYSTEM, "You are german and dont understand any other language. Dont provide any translations.")
    agent = Agent.build(cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B), memory=agent_memory)
    agent.invoke("hello, iam Felix")
    agent.invoke("What did i say my name was?")

    ##### Toolcalling Example

    imaginator_mem = Memory()
    imaginator_mem.add_message(Role.SYSTEM, "You are an advanced, creative AI and always provide suggestions.")
    imaginator = Agent.build(cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B), memory=imaginator_mem)
    answer = imaginator.invoke("You are a Human in a Room. You have no memory. You have to escape. Survive! Wich are the nexz Actions would you like to perform? Keep it short")

    realisator_mem = Memory()
    realisator_mem.add_message(Role.SYSTEM, "You are an advanced, AI. You use body action tools to realise plans of a human brain. Toolcalls only!")
    realisator = Agent.build(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    realisator.register_tools([body_say, perform_body_action])
    realisator.invoke("Human is in a Room. Human has no memory. PLAN_OF_THE_BRAIN:" + answer)

if __name__ == "__main__":
    main()

# test_main.py

from advanced.agent import Agent
import debug
from llm.cache import Cache
from advanced.tool import tool
from llm.memory.simple import SimpleMemory
from llm.model import Model
from llm.runner import Role, Runner

@tool
def get_time() -> str:
    """Return the current system time as an ISO string.

    Returns:
        str: Current system time in ISO 8601 format.
    """
    
    debug.pretty(
        debug.bullet(f"[TOOLCALL] get_time", color=debug.Color.YELLOW),
    )
    return "19:20"


@tool
def skip() -> None:
    """End the agent's turn.

    This tool signals that no further actions are required.

    Returns:
        None
    """
    debug.pretty(
        debug.bullet(f"[TOOLCALL] skip", color=debug.Color.YELLOW),
    )

@tool
def body_say(msg: str) -> str:
    """The human says something something loud.

    Args:
        msg (str): The message text to output.

    Returns:
        str: A confirmation string echoing the supplied message.
    """
    debug.pretty(
        debug.bullet(f"[TOOLCALL] say: {msg}", color=debug.Color.YELLOW),
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
    debug.pretty(
        debug.bullet(f"[TOOLCALL] perform_body_action: {action}", color=debug.Color.YELLOW),
    )
    return f"You did: {action}"

def main():
    cache = Cache()

    # prepare models (so that output is not flooded when LlamaCpp initialises)
    cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)
    cache.get(Model.Local.LlamaCpp.HYBRID_LLAMA3_GROQ_8B_Q8)

    ##### Runner Example

    runner_memory = SimpleMemory()
    runner = cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)
    runner.invoke("hello, im felix", role=Role.USER, memory=runner_memory)
    runner.invoke("What did i say my name was?", role=Role.USER, memory=runner_memory)

    ##### Agent Example

    agent_memory = SimpleMemory()
    agent_memory.add_message(Role.SYSTEM, "You are german and dont understand any other language. Dont provide any translations.")
    agent = Agent.build(cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B), memory=agent_memory)
    agent.invoke("hello, iam Felix")
    agent.invoke("What did i say my name was?")

    ##### Toolcalling Example

    imaginator_mem = SimpleMemory()
    imaginator_mem.add_message(Role.SYSTEM, "You are an advanced, creative AI and always provide suggestions.")
    imaginator = Agent.build(cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B), memory=imaginator_mem)
    answer = imaginator.invoke("You are a Human in a Room. You have no memory. You have to escape. Survive! Wich are the nexz Actions would you like to perform? Keep it short")

    realisator_mem = SimpleMemory()
    realisator_mem.add_message(Role.SYSTEM, "You are an advanced, AI. You use body action tools to realise plans of a human brain. Toolcalls only!")
    realisator = Agent.build(cache.get(Model.Local.LlamaCpp.HYBRID_LLAMA3_GROQ_8B_Q8))
    realisator.register_tools([body_say, perform_body_action])
    realisator.invoke("Human is in a Room. Human has no memory. PLAN_OF_THE_BRAIN:" + answer)

if __name__ == "__main__":
    main()

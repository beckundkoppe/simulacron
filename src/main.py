# test_main.py

from advanced.agent import Agent
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
    print("get_time#")
    return "19:20"


@tool
def skip() -> None:
    """End the agent's turn.

    This tool signals that no further actions are required.

    Returns:
        None
    """
    print("skip#")

@tool
def say(msg: str) -> str:
    """Say something loud.

    Args:
        msg (str): The message text to output.

    Returns:
        str: A confirmation string echoing the supplied message.
    """
    print("say#" + msg)
    return f"You said: {msg}"

@tool
def perform_body_action(action: str) -> str:
    """Perform am action.

    Args:
        action (str): The description of the ction to perform.

    Returns:
        str: A confirmation string.
    """
    print("do#" + action)
    return f"You did: {action}"

def main():
    cache = Cache()

    # prepare models (so that output is not flooded when LlamaCpp initialises)
    cache.get(Model.Local.LlamaCpp.INSTRUCT_MISTRAL_7B)
    cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)

    runner_memory = SimpleMemory()
    runner = cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)
    runner.invoke("hello, im felix", role=Role.USER, memory=runner_memory)
    runner.invoke("What did i say my name was?", role=Role.USER, memory=runner_memory)

    agent_memory = SimpleMemory()
    agent_memory.add_message(Role.SYSTEM, "You are german and dont understand any other language. Dont provide any translations.")
    agent = Agent.build(cache.get(Model.Local.LlamaCpp.INSTRUCT_MISTRAL_7B), memory=agent_memory)
    agent.invoke("hello, iam Felix")
    agent.invoke("What did i say my name was?")

if __name__ == "__main__":
    main()

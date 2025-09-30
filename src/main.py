# test_main.py

from advanced.agent import Agent
import debug.console as console
from llm.cache import Cache
from advanced.tool import tool
from llm.memory.simple import SimpleMemory
from llm.model import Model
from llm.runner import Role, Runner

import inspect

@tool
def body_say(msg: str) -> None:
    """The human says something something loud.

    Args:
        msg (str): The message text to output.
    """
    console.pretty(
        console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {msg}", color=console.Color.YELLOW),
    )

@tool
def perform_body_action(action: str) -> None:
    """Perform a physical action with the human body.

    Args:
        action (str): The description of the ction to perform.
    """
    console.pretty(
        console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {action}", color=console.Color.YELLOW),
    )

def main():
    cache = Cache()

    # prepare models (so that output is not flooded when LlamaCpp initialises)
    ##cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)
    ##cache.get(Model.Local.Ollama.HYBRID_LLAMA3_2_3B)
##
    ####### Runner Example
##
    ##runner_memory = SimpleMemory()
    ##runner = cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B)
    ##runner.invoke("hello, im felix", role=Role.USER, memory=runner_memory)
    ##runner.invoke("What did i say my name was?", role=Role.USER, memory=runner_memory)
##
    ####### Agent Example
##
    ##agent_memory = SimpleMemory()
    ##agent_memory.add_message(Role.SYSTEM, "You are german and dont understand any other language. Dont provide any translations.")
    ##agent = Agent.build(cache.get(Model.Local.Ollama.INSTRUCT_GEMMA3_4B), memory=agent_memory)
    ##agent.invoke("hello, iam Felix")
    ##agent.invoke("What did i say my name was?")
##
    ####### Toolcalling Example
##
    imaginator_mem = SimpleMemory()
    imaginator_mem.add_message(Role.SYSTEM, "You are an advanced, creative AI and always provide suggestions.")
    imaginator = Agent.build(cache.get(Model.Local.LlamaCpp.HYBRID_LLAMA3_GROQ_8B_Q8), memory=imaginator_mem)
    answer = imaginator.invoke("The is Human in a Room. The Human has no memory. The Human have to escape and survive! Wich are the next few Actions you would advice the human to perform? You cant communicate with the yuman, just think!")

    realisator_mem = SimpleMemory()
    imaginator_mem.add_message(Role.SYSTEM, "You use body action tools to realise the suggestions of a human brain. Toolcalls only! You can not communicate with the human but only realise its plans. Dont let the human talk with whith its self")
    realisator = Agent.build(cache.get(Model.Local.LlamaCpp.HYBRID_LLAMA3_GROQ_8B_Q8), memory=imaginator_mem)
    realisator.register_tools([body_say, perform_body_action])
    realisator.invoke("PLANS:" + answer)

if __name__ == "__main__":
    main()

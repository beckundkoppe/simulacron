import current
from enviroment.exception import HardException, SoftException
from enviroment.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from enviroment.world import World
from llm.memory.memory import Role
from util import console


def trycatch(action, success_msg):
    try:
        msg = ""
        message = action()
        if message is None:
            msg = success_msg
        else:
            msg = message

        Success(msg)
        current.RESULT.toolcall_count += 1
    except SoftException as s:
        agent_msg = getattr(s, "agent_message", str(s))
        console_msg = getattr(s, "console_message", agent_msg)
        ActionNotPossible(
            agent_msg,
            console_msg,
            hint=getattr(s, "hint", None),
            context=getattr(s, "context", None),
        )
        current.RESULT.softerror_count += 1
    except HardException as h:
        agent_msg = getattr(h, "agent_message", str(h))
        console_msg = getattr(h, "console_message", agent_msg)
        FormalError(
            agent_msg,
            console_msg,
            hint=getattr(h, "hint", None),
            context=getattr(h, "context", None),
        )
        current.RESULT.harderror_count += 1
    except Exception as e:
        agent_msg = getattr(e, "agent_message", str(e))
        console_msg = getattr(e, "console_message", agent_msg)
        FormalError(
            agent_msg,
            console_msg,
            hint=getattr(e, "hint", None),
            context=getattr(e, "context", None),
        )
        current.RESULT.harderror_count += 1

def check_id(readable_id: str):
    for entity in World.entities:
        if entity.readable_id == readable_id:
            return entity

    agent_entity = getattr(current.AGENT, "entity", None)
    room = agent_entity.room if agent_entity and agent_entity.room else None
    available_ids: list[str] = []
    if room:
        for ent in room.entities:
            if ent and ent.readable_id:
                available_ids.append(ent.readable_id)

    raise HardException(
        f"No object named '{readable_id}' is available in your current room.",
        console_message=(
            f"Lookup failed for '{readable_id}'. Room "
            f"'{room.name if room else 'unknown'}' currently exposes: "
            f"{', '.join(sorted(available_ids)) or 'no interactive objects'}."
        ),
        hint="Check your latest observation for the correct identifier or move closer to the target.",
        context={
            "requested_id": readable_id,
            "room": room.readable_id if room else None,
            "available_ids": sorted(available_ids),
        },
    )

def process_formal_errors(agent) -> bool:
    """Process FormalError results from the result buffer and clear them.
    
    Returns:
        True if any FormalError was found and processed, False otherwise.
    """
    has_error = False
    to_remove = []
    
    for result in Resultbuffer.buffer:
        if not isinstance(result, FormalError):
            continue

        has_error = True
        to_remove.append(result)
        prefix = "[FORMAL ERROR]"
        color = console.Color.RED
        role = Role.SYSTEM

        agent_msg = f"{prefix} {result.agent_message}"
        if result.hint:
            agent_msg = f"{agent_msg} Hint: {result.hint}"

        if agent is not None:
            if agent.memory:
                agent.memory.add_message(role, agent_msg)

        console_lines = [
            console.bullet(
                f"[toolcall] {prefix} {result.console_message}",
                color=color,
            )
        ]
        if result.hint:
            console_lines.append(
                console.bullet(
                    f"\nHint: {result.hint}",
                    color=console.Color.CYAN,
                )
            )
        if result.context:
            console_lines.append(
                console.bullet_multi(
                    f"\nContext: {console.dump_limited(result.context, max_depth=1)}",
                    color=console.Color.BLUE,
                )
            )

        console.pretty(*console_lines, spacing=0)
    
    # Remove processed FormalError results from buffer
    for result in to_remove:
        Resultbuffer.buffer.remove(result)
    
    return has_error

def process_action_results(agent):
    for result in Resultbuffer.buffer:
        if isinstance(result, ActionNotPossible):
            prefix = "[ACTION FAILURE]"
            color = console.Color.RED
            role = Role.USER
        elif isinstance(result, Success):
            prefix = "[ACTION EXECUTED]"
            color = console.Color.YELLOW
            role = Role.USER
        else:
            continue

        agent_msg = f"{prefix} {result.agent_message}"
        if result.hint:
            agent_msg = f"{agent_msg} Hint: {result.hint}"

        if agent.memory:
            agent.memory.add_message(role, agent_msg)

        console_lines = [
            console.bullet(
                f"[toolcall] {prefix} {result.console_message}",
                color=color,
            )
        ]
        if result.hint:
            console_lines.append(
                console.bullet(
                    f"\nHint: {result.hint}",
                    color=console.Color.BLUE,
                )
            )
        if result.context:
            console_lines.append(
                console.bullet_multi(
                    f"\nContext: {console.dump_limited(result.context, max_depth=1)}",
                    color=console.Color.BLUE,
                )
            )

        console.pretty(*console_lines)

def process_results(agent):
    """Process all results from the result buffer and clear it."""
    process_formal_errors(agent)
    process_action_results(agent)
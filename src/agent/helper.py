import current
from enviroment.exception import HardException, SoftException
from enviroment.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from enviroment.world import World
from llm.memory.memory import Role, Type
from util import console


def log_tool_usage(external: bool) -> None:
    """Track tool usage for performance metrics."""
    if not getattr(current, "RESULT", None):
        return

    current.RESULT.toolcall_count += 1
    if external:
        current.RESULT.actions_external += 1
    else:
        current.RESULT.actions_internal += 1


def trycatch(action, success_msg, *, external: bool | None = None):
    # Count tool use upfront; default to internal when category not specified.
    if external is None:
        log_tool_usage(external=False)
    else:
        log_tool_usage(external=external)

    try:
        msg = ""
        message = action()
        if message is None:
            msg = success_msg
        else:
            msg = message

        Success(msg)
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

    entity = current.ENTITY
    room = entity.room if entity and entity.room else None
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

def process_formal_errors(memory=None, collect: bool = False):
    """Process FormalError results from the result buffer and clear them.

    Args:
        memory: Deprecated. The function no longer appends to the supplied
            memory; callers should append returned content themselves.
        collect: When True, also return a list of structured error payloads that
            can be fed back into subsequent prompts (e.g., for retries).

    Returns:
        bool | Tuple[bool, list[dict]]: ``has_error`` if ``collect`` is False,
        otherwise a tuple of ``(has_error, collected_payloads)``.
    """
    has_error = False
    to_remove = []
    collected_payloads = []
    
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

        collected_payloads.append(
            {
                "agent_message": result.agent_message,
                "hint": result.hint,
                "context": result.context,
            }
        )

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
    
    if collect:
        return has_error, collected_payloads

    return has_error

def process_action_results() -> list[tuple[Role, str, Type]]:
    messages: list[tuple[Role, str, Type]] = []
    to_remove = []

    for result in Resultbuffer.buffer:
        to_remove.append(result)

        if isinstance(result, ActionNotPossible):
            prefix = "[ACTION FAILURE]"
            color = console.Color.YELLOW
            role = Role.USER
        elif isinstance(result, Success):
            current.ANY_ACTION = True
            prefix = "[ACTION EXECUTED]"
            color = console.Color.GREEN
            role = Role.USER
        else:
            continue

        agent_msg = f"{prefix} {result.agent_message}"
        if result.hint:
            agent_msg = f"{agent_msg} Hint: {result.hint}"

        messages.append((role, agent_msg, Type.FEEDBACK))

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

        console.pretty(*console_lines)

    for result in to_remove:
        Resultbuffer.buffer.remove(result)

    return messages

def process_results(memory):
    """Process all results from the result buffer and clear it."""
    process_formal_errors(memory)
    for role, msg, msg_type in process_action_results():
        memory.append_message(role, msg, msg_type)

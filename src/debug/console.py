from enum import Enum
import json
import re
import textwrap
from typing import Optional

ansi_re = re.compile(r"\x1b\[[0-9;]*m")

class Color(Enum):
    """Supported ANSI colors"""
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

def strip_ansi(s: str) -> str:
    """Remove ANSI color codes for length calculation."""
    return ansi_re.sub("", s)

def print_colored(text: str, color: Color = Color.RED) -> None:
    """
    Print text in given color to the console using ANSI escape codes.
    """
    # Defensive: ensure color is a Color enum instance
    if not isinstance(color, Color):
        raise ValueError("color must be a Color enum value")
    print(f"{color.value}{text}{Color.RESET.value}")


def _apply_color(text: str, color: Color | None) -> str:
    """Wrap text with ANSI color codes if a color is given."""
    if color is None:
        return text
    if not isinstance(color, Color):
        raise ValueError("color must be a Color enum value or None")
    return f"{color.value}{text}{Color.RESET.value}"


def banner(message: str, char: str = "-", padding: int = 1, color: Color | None = None) -> str:
    """
    Horizontal banner with a line above and below the message.
    """
    line = char * len(message)
    pad = "\n" * padding
    text = f"{pad}{line}\n{message}\n{line}{pad}".strip("\n")
    return _apply_color(text, color)


def box(message: str, char: str = "*", padding: int = 1, color: Color | None = None) -> str:
    """
    Box-style banner framed on all four sides.
    """
    width = len(message) + 4
    border = char * width
    pad = "\n" * padding
    text = f"{pad}{border}\n{char} {message} {char}\n{border}{pad}".strip("\n")
    return _apply_color(text, color)


def title(message: str, underline: str = "=", color: Color | None = None) -> str:
    """
    Title with underline.
    """
    text = f"{message}\n{underline * len(message)}"
    return _apply_color(text, color)


def bullet(message: str, bullet: str = "•", color: Color | None = None) -> str:
    """
    Bullet line. Any newline characters in `message` are removed silently.
    """
    # Replace all types of newline characters with a single space
    single_line = message.replace("\r", " ").replace("\n", " ")
    # Collapse multiple consecutive spaces to one
    single_line = " ".join(single_line.split())

    text = f"{bullet} {single_line}"
    return _apply_color(text, color)

def debug_separator(char: str = "─", length: int = 80, color: Color | None = None) -> str:
    """
    Return a horizontal visual separator line for debug output.

    :param char:   Character used to draw the line (default '─').
    :param length: Visible length of the separator (default 80).
    :param color:  Optional Color enum for ANSI-colored output.
    """
    line = char * length
    return _apply_color(line, color)

def pretty(*parts: str, spacing: int = 1) -> None:
    """
    Combine multiple style parts and print them as one block.

    Args:
        *parts:   strings from banner/box/title/bullet
        spacing:  blank lines inserted between each part
    """
    sep = "\n" * spacing
    print("")
    print(sep.join(part.strip("\n") for part in parts))

def json_dump(object) -> None:
    print(json.dumps(object.__dict__, indent=2, ensure_ascii=False))

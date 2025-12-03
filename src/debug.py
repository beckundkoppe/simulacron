import sys
import io
from contextlib import contextmanager

# debug.py
# Global debug debuguration

VERBOSE_LLAMACPP = False
VERBOSE_LANGCHAIN = False

VERBOSE_LLAMACPPAGENT = False
VERBOSE_LANGCHAIN_TOOL = False

def print_to_file(string: str):
    string = str(string)

    # Append debug output to the configured raw log file (if any).
    import config
    from pathlib import Path

    target = config.APPEND_RAW
    if target is None:
        return

    raw_path = Path(target)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    content = string
    if not content.endswith("\n"):
        content += "\n"

    content += "\n"

    try:
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        # Do not let debug logging break execution.
        pass

@contextmanager
def capture_stdout():
    """Temporarily redirect stdout. English comments only."""
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout
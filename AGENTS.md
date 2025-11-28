# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the engine logic with subpackages such as `agent/`, `enviroment/`, `llm/`, and `util/`. Keep models/data in `data/`, supporting documentation in `doc/`, and reusable assets in `resources/`. Helper scripts (setup, profiling, etc.) live in `script/`. Integration tests and regression checks live in `tests/` alongside the primary `tests/test_*.py` modules, while `README.md`, `requirements.txt`, and `setup.sh` anchor the project root.

## Build, Test, and Development Commands
- `./setup.sh` – boots the environment (select backend, build native bits); run once per machine before other work.
- `./start.sh` – launches the compiled engine once Ollama is up; pairs with a separate `ollama serve` console.
- `ollama serve` – required LLM endpoint; start this in another shell before running `./start.sh`.
- `python -m unittest -v` or `./test.sh` – runs all unit tests after ensuring `venv` and `requirements.txt` dependencies are installed/updated (see `test.sh`’s bootstrap logic).
- `python -m pip install -r requirements.txt` – keep dependencies synced whenever you touch tests or new libraries.

## Coding Style & Naming Conventions
Python modules follow PEP 8: four-space indentation, `snake_case` for functions/variables, and `PascalCase` for classes/entities (e.g., `AgentEntity`). Prefer explicit type hints and `from __future__ import annotations` where backward compatibility matters. Inline comments are sparse; favor docstrings on public classes and descriptive unit-test docstrings. Keep logging and debug helpers centralized in `src/debug/` so formatting tools (currently none mandated) can be applied uniformly.

## Testing Guidelines
Tests live in `tests/` and use the standard `unittest` runner; files begin with `test_` and classes inherit from `unittest.TestCase`. The suite targets environment behavior (movement, containers, connectors), so add new tests near the relevant module (e.g., `enviroment/room.py`). Run `python -m unittest discover tests` or `./test.sh` after editing core logic.

## Commit & Pull Request Guidelines
Commits use sentence-style summaries (see recent history: “Finish reworked agentloop”). Keep messages concise, reference the high-level change, and avoid “WIP.” Pull requests should summarize changes, link related issues or tickets, note testing performed (e.g., `python -m unittest -v`), and include screenshots only when UI artifacts change.

## Configuration & Runtime Notes
- Ollama (>=0.12.1) is required and optionally pairs with CUDA/Vulkan—use the `paru` instructions in `README.md`.
- Selecting a backend happens during `./setup.sh`; re-run the script when switching to a new hardware target.
- Local work expects a Python 3.13.7+ virtual environment in `venv/` (see `test.sh` for creation/activation patterns); commit updates keeping dependencies listed in `requirements.txt`.

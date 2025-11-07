# AGENT INSTRUCTIONS

## Scope
These guidelines apply to the entire repository.

## Project Orientation
- **Entry Point**: `src/main.py` builds the model cache, creates the `Dispatcher`, and runs benchmarks across predefined level configurations.
- **Benchmark Orchestration**: The `Dispatcher` resets levels, preloads models, aggregates timing results, and coordinates each run.
- **Simulation Layer**: `src/game.py` constructs levels, instantiates agents, registers tools, and manages the observation/action loop.
- **Agents & LLM Integration**: `src/advanced/agent.py` contains base classes plus llama.cpp and LangChain implementations, including robust tool-call handling.
- **Environments & Entities**: `src/environment/` hosts level specifications (e.g., `potato.py`) and the entity model in `entity.py`.
- **LLM Catalog & Runners**: `src/llm/model.py` lists available models and backends, while `src/llm/runner.py` implements the execution layer.
- **Roadmap**: Planned extensions live in `doc/TODO.md`.

## Style & Workflow Notes
- Favor modular files over monolithic scripts; separate demo code from production logic.
- Prefer centralized logging utilities over scattered `print` statements.
- Minimize global state mutations—pass configuration explicitly when possible.
- Before committing, consider whether benchmarks or tests are affected and document any commands you execute.

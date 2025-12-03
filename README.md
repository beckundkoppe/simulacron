# Simulacron

<p align="center">
   <img alt="" src="resources/banner.png" width="512" />
</p>

**Simulacron** is an engine that enables large language models to control agents in a simulate interactive world for research purposes.

## Features

* **Multiple backends** – Support for CUDA (build from source), Vulkan, and CPU execution.
* **Flexible architecture** – Supports local and remote hosted LLMs.

## Requirements

* Python >=3.13.7
* Ollama >=0.12.1

## Build & Installation

On Arch-Linux to install Ollama run:

```bash
# AMD-GPU and CPU-only
paru -S ollama

# For NVIDIA GPU acceleration (CUDA)
paru -S ollama ollama-cuda
```

Optional - To install Cuda run:

```bash
paru -S cuda
```

Clone the repository:

```bash
git clone git@github.com:beckundkoppe/simulacron.git
cd simulacron
```

Run the setup script to configure and build:

```bash
./setup.sh
```

During setup you will be prompted to choose a backend. This will look similar to:

```
Available backends:
0) CUDA  (Prebuilt)     [N/A]
1) CUDA  (Build)        [default]
2) Metal (Prebuilt)     [N/A]
3) Metal (Build)        [N/A]
4) HIP                  [N/A]
5) SYCL                 [N/A]
6) OpenBLAS             [N/A]
7) Vulkan
8) CPU
[Enter choice] (10s):
```

Select the desired backend. If unsure, press **Enter** to use the default.

## Running Simulacron

Ollama is now started automatically when a run needs a local Ollama model and
stopped afterwards. If you prefer to keep it running manually you still can:

```bash
# optional
ollama serve
```

Then, inside the simulacron folder, run:
```bash
./start.sh
```

This launches the main executable.

## Benchmarks

1. Configure the benchmark matrix in `src/benchmark/phase_settings.py` (`PHASES`) and per-host allowlists (`RUNNER_CONFIGS`). The order of `allowed_model_teams` matters: runs for the first team are claimed before later entries.
2. Generate TODO lists: `PYTHONPATH=src python -m benchmark.phase_generate` (writes into `results/phase/`).
3. Execute on a runner (optional phase override): `PYTHONPATH=src python -m benchmark.phase_runner [phase]` (reads `results/phase/`, writes `results/runs/`).
4. You can keep `results/` as a separate git repo to sync TODOs/results independently from code.
4. See `doc/distributed_benchmark.md` for multi-host coordination details.

## Repository Structure

```
simulacron/
├─ data/        # Model and embedding files
├─ results/     # Benchmark TODOs and run outputs (can be its own repo)
├─ doc/         # Documentation
├─ ressources/  # Icon and Assets
├─ script/      # Helper scripts
├─ src/         # Core source code
└─ setup.sh     # Run once for setup
└─ start.sh     # Can be used to start the program
```

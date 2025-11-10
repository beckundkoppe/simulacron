# Simulacron

<p align="center">
   <img alt="" src="resources/banner.png" width="512" />
</p>

**Simulacron** is a story engine that uses large language models to simulate interactive worlds and characters for research purposes.

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

Once the setup is complete run Ollama:

```bash
ollama serve
```

In another console, inside the simulacron folder, run:
```bash
./start
```

This launches the main executable.

## Repository Structure

```
simulacron/
├─ data/        # Model and embedding files
├─ doc/         # Documentation
├─ ressources/  # Icon and Assets
├─ script/      # Helper scripts
├─ src/         # Core source code
└─ setup.sh     # Run once for setup
└─ start.sh     # Can be used to start the program
```

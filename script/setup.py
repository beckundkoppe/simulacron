import os
import sys
import subprocess
import time
import shutil
import logging
import platform
import threading
import select
from typing import List, Tuple, Callable

# Import termios and tty only on non-Windows systems
if platform.system() != "Windows":
    import termios
    import tty

# Configuration
GREEN_COLOR = "\033[1;32m"
ORANGE_COLOR = "\033[1;33m"
RED_COLOR = "\033[1;31m"
NORMAL_COLOR = "\033[0m"
RECOMMENDED_TEXT = "[default]"
UNAVAILABLE_TEXT = "[N/A]"
TIMEOUT = 10                    # Countdown duration in seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Backend configurations: (priority, name, install command, detection function)
BACKENDS = [
    (
        "5",
        "CUDA\t (Prebuilt)",
        'pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121',
        lambda: (
            shutil.which("nvidia-smi") 
            and subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0 
            and platform.python_version_tuple()[:2] in [("3","10"),("3","11"),("3","12")]
            and "CUDA Version" in subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout
        )
    ),
    (
        "10",
        "CUDA\t (Build)",
        'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
            shutil.which("nvidia-smi") 
            and shutil.which("nvcc") 
            and subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0
        )
    ),
    (
        "15",
        "Metal (Prebuilt)",
        'pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal',
        lambda: (
    platform.system() == "Darwin"
    and "arm64" in platform.machine().lower()
    and float(platform.mac_ver()[0].split(".")[0]) >= 11
    and platform.python_version_tuple()[:2] in [("3","10"),("3","11"),("3","12")]
)

    ),
    (
        "20",
        "Metal (Build)",
        'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
    platform.system() == "Darwin" 
    and "arm64" in platform.machine().lower()
    and float(platform.mac_ver()[0].split(".")[0]) >= 11
)

    ),
    (
        "30",
        "HIP",
        'CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
    shutil.which("rocminfo") 
    and subprocess.run("rocminfo", shell=True, capture_output=True).returncode == 0
    and "AMD" in subprocess.run("rocminfo", shell=True, capture_output=True, text=True).stdout
)
    ),
    (
        "40",
        "SYCL",
        'source /opt/intel/oneapi/setvars.sh && CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
    shutil.which("icx") 
    and shutil.which("icpx") 
    and subprocess.run("sycl-ls", shell=True, capture_output=True).returncode == 0
)
    ),
    (
        "50",
        "OpenBLAS",
        'CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
    subprocess.run("pkg-config --exists openblas", shell=True).returncode == 0
    or os.path.exists("/usr/lib/libopenblas.so")
    or os.path.exists("/opt/homebrew/opt/openblas/lib/libopenblas.dylib")
)
    ),
    (
        "60",
        "Vulkan",
        'CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: (
    shutil.which("vulkaninfo")
    and subprocess.run("vulkaninfo --summary", shell=True, capture_output=True).returncode == 0
)

    ),
    (
        "70",
        "CPU",
        'pip install llama-cpp-python --no-cache-dir --force-reinstall',
        lambda: True  # CPU is always available
    ),
]

def check_backend_availability() -> List[Tuple[int, str, str, bool]]:
    """Check which backends are available.
    
    Returns:
        List of tuples: (priority, name, install_command, available)
    """
    results = []
    for prio, name, cmd, detect in BACKENDS:
        try:
            available = bool(detect())
        except Exception:
            available = False
        results.append((int(prio), name, cmd, available))
    return results

def detect_best_backend(available_backends: List[Tuple[int, str, str, bool]]) -> int:
    """Detect best backend index (lowest prio with available=True)."""
    sorted_backends = sorted(enumerate(available_backends), key=lambda x: x[1][0])
    for idx, (_, _, _, available) in sorted_backends:
        if available:
            return idx
    return len(available_backends) - 1  # CPU fallback


def display_menu(best_index: int, available_backends: List[Tuple[int, str, str, bool]]) -> None:
    """Display the backend selection menu with availability indicators."""
    print(f"{ORANGE_COLOR}Available backends:{NORMAL_COLOR}")
    for i, (prio, name, _, available) in enumerate(available_backends):
        if i == best_index:
            print(f"{GREEN_COLOR}{i}) {name:20} {RECOMMENDED_TEXT} {NORMAL_COLOR}")
        elif available:
            print(f"{i}) {name:20}")
        else:
            print(f"{RED_COLOR}{i}) {name:20} {UNAVAILABLE_TEXT}{NORMAL_COLOR}")

def get_user_choice(best_index: int, available_backends: List[Tuple[int, str, str, bool]]) -> int:
    """Get user choice (by index) with countdown timer."""
    time_left = TIMEOUT
    choice = ""
    stop_countdown = threading.Event()

    indices = [str(i) for i in range(len(available_backends))]
    prompt = f"[Enter choice]"

    def countdown():
        nonlocal time_left
        while time_left > 0 and not choice and not stop_countdown.is_set():
            print(f"\r{prompt} ({time_left}s): ", end="", flush=True)
            time.sleep(1)
            time_left -= 1

    countdown_thread = threading.Thread(target=countdown, daemon=True)
    countdown_thread.start() 

    try:
        while time_left > 0 and not choice and not stop_countdown.is_set():
            if platform.system() == "Windows":
                import msvcrt
                if msvcrt.kbhit():
                    choice = msvcrt.getch().decode("utf-8", errors="ignore").strip()
            else:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    choice = sys.stdin.readline().strip()

            if choice in indices:
                print()  # newline nach Eingabe
                return int(choice)
            elif choice:
                print(f"\r\033[KInvalid choice. Please enter one of: {', '.join(indices)}.\n", flush=True)
                choice = ""
                time_left = TIMEOUT

        # Timeout oder Abbruch → Default
        if not choice and not stop_countdown.is_set():
            print(f"\nUsing default [{GREEN_COLOR}{best_index}{NORMAL_COLOR}].")
            return best_index
        return best_index
    except KeyboardInterrupt:
        stop_countdown.set()
        raise
    finally:
        stop_countdown.set()
        countdown_thread.join(timeout=0.1)


def install_llama_cpp(choice: int, available_backends: List[Tuple[int, str, str, bool]]) -> None:
    """Install llama-cpp-python with chosen backend."""
    try:
        prio, name, install_cmd, available = available_backends[choice]
    except IndexError:
        logger.error("Invalid backend index. Exiting.")
        sys.exit(1)

    if not available:
        logger.warning(f"Selected {name} is marked as unavailable. Installation may fail.")

    logger.info(f"Installing llama-cpp-python with {name} support")
    try:
        subprocess.check_call(install_cmd, shell=True)
        logger.info(f"Successfully installed llama-cpp-python with {name} support")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install llama-cpp-python with {name} support: {e}")
        sys.exit(1)


def main():
    """Main function to orchestrate the installation process."""
    try:
        # Check backend availability
        available_backends = check_backend_availability()

        # Detect best backend (liefert Index)
        best_index = detect_best_backend(available_backends)

        # Display menu
        display_menu(best_index, available_backends)

        # Get user choice (Index oder Default)
        choice = get_user_choice(best_index, available_backends)

        # Install selected backend
        install_llama_cpp(choice, available_backends)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
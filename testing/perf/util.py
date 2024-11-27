import os
import platform
import subprocess
import time
from typing import Callable


def get_rpi_version():
    try:
        cmd = ["cat", "/proc/device-tree/model"]
        model = subprocess.check_output(cmd).decode('utf-8').rstrip('\x00').strip()

        if 'Pi 5' in model:
            return 'rpi-5'
        elif 'Pi 4' in model:
            return 'rpi-4'
        elif 'Pi 3' in model:
            return 'rpi-3'
        else:
            return 'non-rpi'
    except FileNotFoundError:
        return 'non-rpi'
    except subprocess.CalledProcessError:
        return 'Error reading Raspberry Pi version.'


def print_platform_info():
    print("=" * 70)
    print("System:", platform.system())  # e.g., 'Linux', 'Windows', 'Darwin' for macOS
    print("Node Name:", platform.node())
    print("Release:", platform.release())
    print("Version:", platform.version())
    print("Machine:", platform.machine())  # e.g., 'x86_64'
    print("Processor:", platform.processor())  # Processor name
    print("Architecture:", platform.architecture())  # e.g., ('64bit', 'ELF')
    print("Raspberry Pi Version:", get_rpi_version())
    print("=" * 70)


def benchmark(func: Callable, max_runs: int = 1000, time_limit: int = 5, test_name: str = "func", *args, **kwargs):
    """
    Test the performance of a given function.
    Prints total runtime & average duration per call (both in milliseconds).

    Args:
        func: The function to test.
        max_runs: Maximum number of times to run the function.
        time_limit: Maximum time (in seconds) to spend running the function.
        test_name:
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """
    start_time = time.time()
    run_count = 0

    for _ in range(max_runs):
        current_time = time.time()

        func(*args, **kwargs)
        run_count += 1

        if current_time - start_time >= time_limit:
            break

    total_runtime = (time.time() - start_time) * 1000
    average_duration = total_runtime / run_count

    print(f"[{test_name}]\tTotal runtime: {total_runtime:.0f}ms   nRuns: {run_count}    {average_duration:.3f}ms/call")

    return total_runtime, average_duration


def add_dll_dirs():
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            os.add_dll_directory(directory)
        except:
            pass

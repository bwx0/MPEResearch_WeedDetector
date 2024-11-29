import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

"""
If you want to do it yourself:
mkdir build && cd build
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make install -j
"""


def build_perf_cpp(force_rebuild: bool):
    os.chdir(Path(Path(__file__).parent, "cpp"))
    print(f"Current working directory: {os.getcwd()}")
    if force_rebuild:
        # remove the entire build directory
        shutil.rmtree("build", ignore_errors=True)

    # run CMake if the build directory does not exist
    if not os.path.exists("build"):
        Path("build").mkdir(parents=True)
        subprocess.run(["cmake",
                        "-S", ".",
                        "-B", "build",
                        "-G", "Unix Makefiles",
                        # r"-Dpybind11_DIR='../../../venv/Lib/site-packages/pybind11/share/cmake/pybind11'",
                        # r"-DOpenCV_DIR='D:\OpenCV\opencv_4_10_0\local_build'",
                        "-DCMAKE_BUILD_TYPE=Release"],
                       stdout=sys.stdout,
                       stderr=sys.stderr,
                       check=True)
    os.chdir(Path(Path(__file__).parent, "cpp", "build"))
    subprocess.run(["make", "install", "-j"],
                   stdout=sys.stdout,
                   stderr=sys.stderr,
                   check=True)


def main():
    parser = argparse.ArgumentParser(description="Compile and install the C++ part of the perf test.")

    parser.add_argument(
        "force_rebuild", type=int, help="Rebuild from scratch.", default=0, nargs="?"
    )

    args = parser.parse_args()

    build_perf_cpp(force_rebuild=bool(args.force_rebuild))


if __name__ == "__main__":
    main()

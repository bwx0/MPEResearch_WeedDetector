import os
import shutil
import subprocess
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
        print(subprocess.check_output(["cmake",
                                       "-S", ".",
                                       "-B", "build",
                                       "-G", "Unix Makefiles",
                                       # r"-Dpybind11_DIR='../../../venv/Lib/site-packages/pybind11/share/cmake/pybind11'",
                                       # r"-DOpenCV_DIR='D:\OpenCV\opencv_4_10_0\local_build'",
                                       "-DCMAKE_BUILD_TYPE=Release"]).decode('utf-8'))
    os.chdir(Path(Path(__file__).parent, "cpp", "build"))
    print(subprocess.check_output(["make", "install", "-j"]).decode('utf-8'))


if __name__ == "__main__":
    build_perf_cpp(force_rebuild=False)
    # build_perf_cpp(force_rebuild=True)

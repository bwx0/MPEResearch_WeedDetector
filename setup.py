import os
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

module_name="roiyolowd"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('CMake must be installed to build the following extensions: ' +
                               ', '.join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import pybind11
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Release'

        # CMake arguments
        cmake_args = [
            f"-GUnix Makefiles",
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPYBIND11_INCLUDE_DIR={pybind11.get_include()}',
            f'-DCMAKE_BUILD_TYPE={cfg}'
        ]

        # Build arguments
        build_args = ['--config', cfg]
        build_args += ['--', '-j']

        # Build temporary directory
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)

        # Configure and build the extension
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'native_reassembler_module'] + build_args, cwd=build_temp)

setup(
    name=module_name,
    version='0.1.0',
    author='Wx',
    author_email='----',
    description='YOLOv8-based weed detector incorporated with ROI',
    long_description='',
    ext_modules=[CMakeExtension('native_reassembler_module.native_reassembler_module', sourcedir='./native_reassembler_module/cpp')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    packages=find_packages(include=[module_name, f"{module_name}.*"]),
    # setup_requires=['pybind11==2.13.6', 'cmake==3.25.0'],  # moved to requirements.txt
    include_package_data=True,
)

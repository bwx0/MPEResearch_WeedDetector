# Performance Testing

## Overview

This folder includes a few scripts for benchmarking the execution speed of 3 tasks:
- Calculating ExGI values for an RGB image and thresholding based on a lower bound.
- Converting an RGB image to HSV format and thresholding based on lower & upper bounds for H, S, and V.
- Converting an RGB image to HSV format.

These tasks are implemented using three different approaches:
- Implementing in Python
- Implementing in C++
- A combination of Python and C++ via Pybind11

## Directory Structure
```
└── 📁 perf/
    ├── 📁 cpp/
    │   ├── 📁 build/                               make_perf_cpp.py uses this folder to build 
    │   ├── 🖼️ 1080p.png                            Test image
    │   ├── 📜 bindings.cpp                         pybind11 binding entries
    │   ├── 📝 CMakeLists.txt                       
    │   ├── 📜 exgi.cpp                             ExGI-based binarization implementations
    │   ├── 📜 exgi.h                               
    │   ├── 📜 exgi_test.cpp                        A program entry
    │   ├── 📜 hsv.cpp                              HSV-based binarization implementations
    │   ├── 📜 hsv.h                                
    │   ├── 📜 hsv_test.cpp                         A program entry
    │   └── 📜 inplace_test.cpp                     Inplace vs non-inplace thresholding test
    ├── 🖼️ 1080p.png                                Test image
    ├── 🖼️ 1200x800.png                             Test image
    ├── 📜 make_perf_cpp.py                         Run this script to build the cpp module. Or do it yourself by running the cmds in the script.
    ├── 🗃️ perf_cpp.cp310-win_amd64.pyd             The built module will be placed here.
    ├── 📜 perf_cpp.pyi                             Cpp module stub
    ├── 📝 README.md                                
    ├── 📜 util.py                                  Get platform info, test avg runtime of a function
    ├── 📜 vegetation_indices.py                    Compare runtime of different vegetation indices.
    ├── 📜 vi.py                                    Check if pybind impls and python impls produce the same result.
    ├── 📜 vi_pybind.py                             Compare runtime of different pybind impls of VI-based binarization.
    └── 📜 vi_python.py                             Compare runtime of different python impls of VI-based binarization.
```

# Main Findings
Detailed results can be found in the `results` folder.


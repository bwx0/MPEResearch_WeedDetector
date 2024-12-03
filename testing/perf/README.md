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
    │   ├── 📁 build/                       make_perf_cpp.py uses this folder to build 
    │   ├── 🖼️ 1080p.png                    Test image
    │   ├── 📜 bindings.cpp                 pybind11 binding entries
    │   ├── 📝 CMakeLists.txt               
    │   ├── 📜 exgi.cpp                     ExGI-based binarization implementations
    │   ├── 📜 exgi.h                       
    │   ├── 📜 exgi_test.cpp                A program entry
    │   ├── 📜 hsv.cpp                      HSV-based binarization implementations
    │   ├── 📜 hsv.h                        
    │   ├── 📜 hsv_test.cpp                 A program entry
    │   └── 📜 inplace_test.cpp             Inplace vs non-inplace thresholding test
    ├── 🖼️ 1080p.png                        Test image
    ├── 🖼️ 1200x800.png                     Test image
    ├── 📜 make_perf_cpp.py                 Run this script to build the cpp module. Or do it yourself by running the cmds in the script.
    ├── 🗃️ perf_cpp.cp310-win_amd64.pyd     The built module will be placed here.
    ├── 📜 perf_cpp.pyi                     Cpp module stub
    ├── 📝 README.md                        
    ├── 📜 util.py                          Get platform info, test avg runtime of a function
    ├── 📜 vegetation_indices.py            Compare runtime of different vegetation indices.
    ├── 📜 vi.py                            Check if pybind impls and python impls produce the same result.
    ├── 📜 vi_pybind.py                     Compare runtime of different pybind impls of VI-based binarization.
    └── 📜 vi_python.py                     Compare runtime of different python impls of VI-based binarization.
```

# Results
Detailed execution time data can be found in the `results` folder.

### Vegetation Index & Binarisation
One of the most important steps in both OWL and my project is calculating vegetation index (e.g. ExGI and HSV-based ones) for an image and binarise it.

I tried a couple of possible ways to do that, and these are the execution times of the optimal implementations.


**Table: Execution time (ms) of 3 tasks using 3 different approaches on a Raspberry Pi 5**

| Task                | Pure C++       | Python (with OpenCV) | Python & Pybind11 & C++ |
|---------------------|----------------|----------------------|-------------------------|
| RGB2HSV             | 0.64           | 2.46                 | 3.69                    |
| ExGI & Binarisation | 0.72           | 13.94                | 0.91                    |
| HSV & Binarisation  | 5.46           | 11.55                | 5.46                    |

The interoperation between Python and C++ introduces an overhead of around 0.2ms,
based on the data above and some other experiments.

### YOLOv8 performance with different `imgsz` during training vs inferencing

**Table: Speed & accuracy with different `imgsz` during training and inferencing**

| Training imgsz | Inferencing imgsz  | mAP50 (YOLOv8) | mAP50 (YOLOv8 + ROI) | Detection frame rate | Inference Time |
|----------------|--------------------|---------------:|---------------------:|---------------------:|---------------:|
| 480            | 480                |          32.6% |                72.9% |             18.4 fps |          43 ms |
| 480            | 640                |          59.4% |                75.6% |              6.1 fps |          90 ms |
| 640            | 480                |          20.0% |                68.5% |              4.0 fps |          47 ms |
| 640            | 640                |          32.6% |                76.7% |              8.7 fps |          93 ms |

- Measured on a Raspberry Pi 5.
- Inference time is the time reported by the YOLOv8 library.
- Detection frame rate is the overall detection frame rate observed by the user.
- Detection frame rate is lower than the reciprocal of inference time due to overhead from post-processing. Refer to library logs for more details.
- It is quite strange that while smaller imgsz can significantly speed up inferencing (at the cost of some accuracy), a mismatch between the `imgsz` during training and inferencing can cause a huge overhead in postprocessing.





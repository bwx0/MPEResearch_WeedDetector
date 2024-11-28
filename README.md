# In-Crop Weed Detection through YOLOv8 and ROI Extraction

## Background

[OpenWeedLocator](https://github.com/geezacoleman/OpenWeedLocator):  An open-source, low-cost device for fallow weed
detection. The software runs on a Raspberry Pi.

## Introduction

### The Problem

- YOLOv8 has a fixed input image size of 640x640, which is relatively small.
- Higher-resolution images (e.g. 1080p) have to scale down to fit, leading to a loss of detail and a decrease in
  detection accuracy.
- Weeds can be small in images, which become even harder to detect after scaling.
- [Using a sliding window](https://docs.ultralytics.com/guides/sahi-tiled-inference/) can address the issue. But
  performing inference multiple times on the same image is too computationally expensive for realtime weed detection.

### The Solution

This project focuses on scenarios like crops in their early growth stages, where plants are sparse and can be easily
distinguished from the background based on color. The approach involves:

- Detecting and extracting bounding boxes for green regions (ROIs), which are areas that likely contain plants.
- Assemble these ROIs into a compact, square-like image.
- Performing YOLOv8 object detection on the reassembled image.
- Mapping the detection results back to the original image.

The reassembled image is smaller, but it keeps most of the areas where plants are present.
Therefore, the scaling is less intensive, keeping more resolution and details,
leading to better detection accuracy without relying on sliding window techniques.

### Challenges

This project is indented to run on a Raspberry Pi 5, which has limited computational power.
I can't decide how fast the YOLOv8 can detect objects for now,
but I can make the other preprocessing steps (ROI extraction and reassembling) as fast as possible.

### What this project can do

This project primarily includes:

- A fast vegetation index-based ROI extractor.
- A reassembler to organize ROIs into a compact image.
- A `WeedDetector` class that integrates all components. Simply provide an image, and it will return a list of detected
  weeds.
- Additional tools for preparing datasets to fine-tune YOLOv8.

### Results

- Preprocessing a 1080p image takes ~5ms on a Raspberry Pi 5.
- Significantly improves in-crop weed detection accuracy. (But our labelled dataset is quite small and not good quality, so we'll
  need more labelled data to get a more useful and reliable result)

## Project Structure

```
📁 roiyolowd/
├── 📁 cpptest/       Some legacy tests
├── 📁 dataset/       Exported datasets from LabelStudio and generated/processed datasets for fine-tuning and evaluation.
├── 📁 models/        Place fine-tuned models in this folder
├── 📁 native_reassembler_module/  The ROI reassembler implementing in C++ and packaging as a Python module.
├── 📁 plots/         Scripts that produce plots for my thesis.
├── 📁 roiyolowd/     The main module.
├── 📁 test_data/     Test images and videos.
└── 📁 yolo_tools/    A few utilities for YOLOv8, including the script that prepare dataset for fine-tuning and evaluation.
```

## Build & Run

### Create a virtual environment

```bash
python -m venv venv
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements.txt --system-site-packages # if you want to reuse existing packages
```

### Build and install the project

Install the project as an editable module, so that modified Python code takes effect without having to reinstall.

```bash
pip install -e .
pip install .  # Do normal installation if you want to install for other projects
```

Please remove `project_folder/build` before installing, or it will fail. But I have no idea why this happens.

You can also manually build the native module.
This become useful if you are modifying the C++ code, as it allows you to quickly update the module without having to
reinstall the entire project.
These are some commands you will need:

```bash
cd native_reassembler_module/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install -j
```

## Troubleshooting

### OpenCV error when using `cv2.VideoCapture()` to read a video

```
[ WARN:0@9.573] global cap_ffmpeg_impl.hpp:1541 grabFrame packet read max attempts exceeded,
 f your video have multiple streams (video, audio) try to increase attempt limit by setting
 environment variable OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096).
```

Solution: Set the following environment variable using bash or in your IDE.

```bash
export OPENCV_FFMPEG_READ_ATTEMPTS = 10000000
```

### YOLOv8 inference is much slower than the expected 100ms/img

I noticed YOLOv8 was taking around 240ms to detect objects in an image,
which is way slower than the claimed 10 FPS on the RPi 5.
After some digging, I managed to fix it by switching `torchvision` to version 0.20.1.

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

Key idea: Opt out non-weed regions by relatively simple traditional algorithm, and let the
deep learning model focus only on the remaining part of the image.

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
├── 📁 native_reassembler_module/  The ROI reassembler implemented in C++ and packaged as a Python module.
├── 📁 plots/         Scripts that produce plots for my thesis.
├── 📁 roiyolowd/     The main module.
├── 📁 test_data/     Test images and videos.
└── 📁 yolo_tools/    A few utilities for YOLOv8, including the script that prepare dataset for fine-tuning and evaluation.
```

The `native_reassembler_module` is a C++ implementation of the preprocessing steps described above.
While a Python implementation is also available in the main `roiyolowd` module, it is much slower.

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


## Fine-tune YOLOv8
- Step 1: Label your images in [LabelStudio](https://labelstud.io/).
- Step 2: Export the project in the YOLO format.
- Step 3: Use `yolo_tools/yolo_slicer.py` to augment and convert the exported dataset into one that can directly be used by YOLOv8.
- Step 4: Use `yolo_tools/yolo_train.py` to train the model with the prepared dataset (the one named `{your_dataset_name}_final`).

To improve the frame rate, you can lower the `imgsz` for both training and inference,
but this comes at the expense of reduced *detection capacity*, in the sense that
a reassembled image of size 500x500 can maintain its full detail (without scaling)
when used with `imgsz=640`. However, it will require scaling if `imgsz=480`. 
If you know the target environment has sparse vegetation (fewer green areas), you can safely reduce `imgsz` to
achieve higher frame rates without significant drawbacks.

## Add your ROI extractor

Create your ROI extractor by extending the `ROIExtractor` class.

For now, I assume that ROI extractors are stateless and rely on traditional computer vision techniques,
such as identifying potential weeds based on color.

The point is, ROI extractors should opt out plants or regions that are confidently identified as non-weeds by the 
traditional algorithm, and leave the rest for the deep learning model for further verification.

## Troubleshooting

### OpenCV error when using `cv2.VideoCapture()` to read a video

```
[ WARN:0@9.573] global cap_ffmpeg_impl.hpp:1541 grabFrame packet read max attempts exceeded,
 f your video have multiple streams (video, audio) try to increase attempt limit by setting
 environment variable OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096).
```

Solution: Set the following environment variable using bash or through your IDE.

```bash
export OPENCV_FFMPEG_READ_ATTEMPTS = 10000000
```

### YOLOv8 inference is much slower than the expected 100ms/img

I noticed YOLOv8 was taking around 240ms to detect objects in an image,
which is way slower than the claimed 10 FPS on the RPi 5.
After some digging, I managed to fix it by switching `torchvision` to version 0.20.1.

## Some Other Ideas

When dealing with a large number of plants, which results in a very large reassembled image,
we can skip preprocessing and feed the original image directly into the detection model.
Alternatively, we can randomly discard some ROIs from the current frame so that the model can
still accurately detect a portion of the weeds while leaving the rest for the future. That way,
a weed still have a decent chance to be detected at some point (not every time it shows up in the image),
given that the camera is moving moderately slow.

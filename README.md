
```
[ WARN:0@9.573] global cap_ffmpeg_impl.hpp:1541 grabFrame packet read max attempts exceeded,
 f your video have multiple streams (video, audio) try to increase attempt limit by setting
 environment variable OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096).
```

```bash
export OPENCV_FFMPEG_READ_ATTEMPTS = 10000000
```

Create a virtual environment

```bash
python -m venv venv
```

Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements.txt --system-site-packages # if you want to reuse existing packages
```

Build and install the project

```bash
pip install -e .
pip install -e . --no-build-isolation
```

Quick Build
```bash
cd native_reassembler_module/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install -j
```
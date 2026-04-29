# Standalone COLMAP Training CLI

This project builds only the headless training path needed to train a splat from an existing COLMAP dataset. It reuses LichtFeld's `core`, `io`, and `training` targets, but skips the GUI app, visualizer, MCP server, and Python bindings.

## Option A: Export A Self-Contained Repo

From the full LichtFeld checkout on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\projects\train_colmap_headless\export_option_a.ps1 C:\path\to\lichtfeld-colmap-trainer
```

Then make that destination a new repo:

```powershell
cd C:\path\to\lichtfeld-colmap-trainer
git init
git add .
git commit -m "Initial standalone COLMAP training build"
```

The exported repo contains:

```text
CMakeLists.txt
README.md
vcpkg.json
tools/train_colmap_headless.cpp
stubs/python_runner_stub.cpp
cmake/
external/
src/core/
src/geometry/
src/io/
src/training/
src/python/runner.hpp
```

## What It Builds

- `train_colmap_headless`: minimal CLI entrypoint from `tools/train_colmap_headless.cpp`
- `lfs_core`: scene, tensors, parameters, image loading hooks
- `lfs_io`: COLMAP loader plus the repo's existing IO target
- `lfs_training`: LichtFeld/Inria-style Gaussian training loop and CUDA rasterizers
- `lfs_python_utils`: local stub, only to satisfy the optional trainer Python-script hook

## Colab Build Cells

From a fresh clone of the exported repo:

```bash
sudo apt-get update
sudo apt-get install -y git curl zip unzip tar pkg-config ninja-build cmake build-essential nasm autoconf automake autoconf-archive libtool libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

```bash
git clone https://github.com/microsoft/vcpkg.git /content/vcpkg
/content/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=/content/vcpkg
```

```bash
cmake -S . -B build_train_cli -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build_train_cli --target train_colmap_headless -j2
```

## Run

Point `--data-path` at the folder that contains your COLMAP output and images, then choose an output folder:

```bash
./build_train_cli/train_colmap_headless \
  --data-path /content/data/my_scene \
  --output-path /content/out \
  --iter 30000
```

The exact dataset layout accepted here is the same one used by `src/io/loaders/colmap_loader.cpp` and `src/training/training_setup.cpp`.

## Notes

When this folder is used inside the full LichtFeld checkout, CMake may create local symlinks named `src`, `external`, and `cmake` so it can reuse the parent repo sources. The Option A export does not need those symlinks because the copied repo already contains those folders.

The current `lfs_io` target still brings in the repo's broad IO dependencies, including video, USD, mesh, WebP, archive, OpenImageIO, and nvImageCodec support. If Colab dependency setup becomes too slow, the next step is to split a smaller `lfs_io_colmap` target containing only `loader.cpp`, `formats/colmap.cpp`, `loaders/colmap_loader.cpp`, image decode/cache, and checkpoint loading.
"# c-train-litchfeild" 

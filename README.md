# Edge-Enhancing Bi-Histogram Equalization (Final)

This project implements an edge-enhancing bi-histogram equalization pipeline with:
- C++ kernels exposed to Python via Pybind11 (`he_core`)
- OpenMP acceleration and runtime thread control
- Unified image/video/webcam processing in Python
- Benchmark and visualization utilities
- FastAPI dashboard backend with SSE streaming and browser frontend

The current codebase is aligned to the paper-oriented pipeline used in your final version, including dynamic `mu` from guided-filter output and `kappa` control from CLI.

## Project Structure

```text
CV project/
├── CMakeLists.txt
├── build_he_core.ps1
├── src/
│   ├── core.cpp
│   ├── common.h
│   ├── member1/
│   ├── member2/
│   ├── member3/
│   └── test.cpp
├── python/
│   └── pipeline.py
├── benchmark_omp.py
├── generate_visuals.py
├── server.py
├── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── outputs/
└── dataset/
        ├── images/
        └── videos/
```

## Prerequisites

- Python 3.10+
- CMake 3.15+
- MSYS2 UCRT64 MinGW toolchain on Windows:
    - `C:/msys64/ucrt64/bin/g++.exe`
    - `C:/msys64/ucrt64/bin/gcc.exe`

## Setup Commands (Windows PowerShell)

Run from project root:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pybind11
```

## Build Commands (Recommended: One-File Automation)

Use the provided script that handles cache issues and validates import:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_he_core.ps1
```

Useful variants:

```powershell
# Skip reinstalling python packages
powershell -ExecutionPolicy Bypass -File .\build_he_core.ps1 -SkipInstall

# Force clean rebuild
powershell -ExecutionPolicy Bypass -File .\build_he_core.ps1 -Clean
```

## Build Commands (Manual Alternative)

```powershell
& .\.venv\Scripts\Activate.ps1
$VENV_PY = (Resolve-Path .\.venv\Scripts\python.exe).Path
$PYBIND11_DIR = & $VENV_PY -m pybind11 --cmakedir

if (Test-Path build) { Remove-Item -Recurse -Force build }

cmake -S . -B build -G "MinGW Makefiles" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER="C:/msys64/ucrt64/bin/gcc.exe" `
    -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" `
    -Dpybind11_DIR="$PYBIND11_DIR" `
    -DPYTHON_EXECUTABLE="$VENV_PY"

cmake --build build --config Release -j 4
```

## Rebuild After C++ Changes

If you edit `src/core.cpp`, only rebuild (no reconfigure needed):

```powershell
cmake --build build --config Release -j 4
```

Re-run full configure only if:
- `CMakeLists.txt` changed
- toolchain/compiler changed
- pybind11/python path changed
- `build/` was deleted

## Run Commands

### 0) Interactive Dashboard (FastAPI + Frontend)

Start the dashboard API server:

```powershell
uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

Open in browser:

```text
http://127.0.0.1:8001/frontend/
```

Dashboard endpoints:
- `POST /process/image`
- `POST /process/video` (SSE stream)
- `GET /webcam/stream` (SSE stream)
- `POST /webcam/stop`
- `POST /benchmark`
- `GET /health`

Video outputs are written to:

```text
outputs/<video_name>/
```

The dashboard saves:
- original processed video
- enhanced processed video
- diff heatmap video

These files are served back to the frontend from `/outputs/...` for final playback after processing completes.

### 1) Main Pipeline (Image)

```powershell
python python/pipeline.py dataset/images/4.1.03.tiff
```

With diff panel:

```powershell
python python/pipeline.py dataset/images/4.1.05.tiff --show-diff --diff-gain 6 --threads 4
```

### 2) Main Pipeline (Video File)

```powershell
python python/pipeline.py dataset/videos/Q001.mp4 --threads 4
```

With stronger sharpening:

```powershell
python python/pipeline.py dataset/videos/M001.mp4 --kappa 5.0 --show-diff --diff-gain 6 --threads 4
```

### 3) Main Pipeline (Webcam)

```powershell
python python/pipeline.py 0 --threads 4
```

### 4) OpenMP Benchmark

```powershell
python benchmark_omp.py --video dataset/videos/Q001.mp4 --frames 100
```

### 5) Visualization Figures

```powershell
python generate_visuals.py --image dataset/images/5.1.11.jpg
```

Expected output files:
- `bi_histogram_visual.png`
- `busyness_map_visual.png`
- `enhanced_for_visual_report.png`

## Pipeline CLI Reference

```text
python python/pipeline.py <input> [--kappa K] [--show-diff] [--diff-gain G] [--threads N]
```

- `<input>`: image path, video path, or `0` for webcam
- `--kappa`: sharpening strength (`default: 5.0`)
- `--show-diff`: display third panel showing absolute-difference heatmap
- `--diff-gain`: diff-map amplification (`default: 5.0`)
- `--threads`: OpenMP thread count (`default: 4`)

## Troubleshooting

### `No module named 'he_core'`

Rebuild module:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_he_core.ps1 -Clean
```

### `Python libraries not found` during CMake configure

You are likely using the wrong Python executable in cache. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_he_core.ps1 -Clean
```

### `DLL load failed while importing he_core`

The pipeline already adds `C:\msys64\ucrt64\bin` at runtime. If needed, ensure this folder exists and MSYS2 UCRT64 is installed.

### Output video plays too fast

Current pipeline writes with source FPS and normal playback speed. Rebuild and rerun if using an old binary/script combination.

## Final Notes

- Use `build_he_core.ps1` as the default build entrypoint for consistency.
- For day-to-day `core.cpp` edits, rebuild only.
- Keep dataset media under `dataset/images` and `dataset/videos` and run commands exactly as shown above.
- For dashboard video mode, verify `outputs/` is writable so final playback files can be generated and served.

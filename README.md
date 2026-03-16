# Edge-Enhancing Bi-Histogram Equalisation using Guided Image Filter
### Week 1 — C++ / Python Hybrid Implementation

This project implements the paper in three modular components, one per team member.
Each member folder contains its own README with full implementation details.

| Member | Folder | Responsibility |
|---|---|---|
| [src/member1/README.md](src/member1/README.md) | `src/member1/` | Colour-space extraction + evaluation metrics |
| [src/member2/README.md](src/member2/README.md) | `src/member2/` | Histogram statistics + adaptive plateau clipping |
| [src/member3/README.md](src/member3/README.md) | `src/member3/` + `python/member3/` | Guided filter coefficients + visualisations |

---

## Architecture

```
CV project/
├── CMakeLists.txt          # Build: pybind11 + OpenMP → he_core.so
├── src/
│   ├── common.h            # Shared types (npy<T> alias, clamp_idx)
│   ├── core.cpp            # Thin pybind11 binder — exposes all 13 functions
│   ├── member1/            # See src/member1/README.md
│   ├── member2/            # See src/member2/README.md
│   └── member3/            # See src/member3/README.md
└── python/
    ├── pipeline.py         # Main driver — calls all C++ functions, Steps 1–7
    └── member3/            # Matplotlib visualisations (PNG output)
```

## Build & Run

### Prerequisites
- CMake 3.10+
- C++17 Compiler (GCC/Clang/MSVC/MinGW) with OpenMP support
- Python 3.8+

### 0. Setup Virtual Environment (Recommended)

#### Windows (PowerShell)
```powershell
# Create and activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (this includes the pybind11 headers needed for compilation)
pip install numpy opencv-python matplotlib pybind11
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python matplotlib pybind11
```

### 1. Build C++ Extension

#### Windows (PowerShell + MinGW)
```powershell
# Create build directory
if (!(Test-Path build)) { mkdir build }

# Configure CMake (Ensure it finds your active Python environment)
# If using a venv, activate it first or set -DPython_EXECUTABLE="Path/To/python.exe"
cmake -S . -B build -G "MinGW Makefiles" -DPYBIND11_FINDPYTHON=ON

# Compile
cmake --build build --config Release
```

#### Linux / macOS / WSL
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

### 2. Run Python Pipeline

```powershell
# From project root
python python/pipeline.py [optional_image_path]
```

**Output files:**
- `week1_gf_visual.png` (Guided Filter maps)
- `week1_distribution_visual.png` (Histogram & CDFs)
- `week2_result.png` (Final Enhanced Image)

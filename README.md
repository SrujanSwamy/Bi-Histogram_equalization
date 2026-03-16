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

## Build & Run (WSL)

```bash
cd "CV project"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../python
python3 pipeline.py ../images/4.2.07.tiff
```

**Output files:** `week1_gf_visual.png`, `week1_distribution_visual.png`

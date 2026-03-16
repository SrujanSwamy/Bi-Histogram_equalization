# Guided Filter Coefficients & Visualisation

## Overview

This module is responsible for the **edge-preservation layer** of the algorithm. A plain bi-histogram equalisation can produce visible ringing or halo artefacts around sharp edges. The **Guided Image Filter (GIF)** solves this by computing per-pixel coefficients that tell Week 2 how aggressively to equalise each location: edge pixels are treated more gently, flat regions get full equalisation.

This module also produces **all visual output** for the project — two PNG figures saved after every run.

The C++ backend uses **OpenMP** to parallelise the coefficient computation. The visualisation code is written in Python with **matplotlib**.

---

## Files

```
src/member3/guided_filter.h      ← C++17 coefficient computation
python/member3/visualize.py      ← Python visualisation
python/member3/__init__.py
```

---

## What This Module Implements

### 1. Guided Filter Coefficients — `compute_gf_coefficients`

The Guided Image Filter is applied in **self-guided** mode: the input image guides itself. For each pixel `k`, a 3×3 window `ω_k` is scanned and two coefficients are computed:

**Local statistics in the window:**

$$\mu_k = \frac{1}{|\omega_k|} \sum_{i \in \omega_k} V_i \quad \text{(local mean)}$$

$$\sigma^2_k = \frac{1}{|\omega_k|} \sum_{i \in \omega_k} V_i^2 - \mu_k^2 \quad \text{(local variance)}$$

**Coefficients:**

$$a_k = \frac{\sigma^2_k}{\sigma^2_k + \varepsilon} \qquad b_k = \mu_k \cdot (1 - a_k)$$

where $\varepsilon = 0.01 \times 255^2 \approx 650$ is a regularisation constant.

**Interpretation of `a_k`:**

| Region type | `σ²` | `a_k` | Meaning |
|---|---|---|---|
| Sharp edge / texture | High | → 1.0 | Preserve the original signal — don't over-equalise |
| Flat / smooth area | Low | → 0.0 | Allow full equalisation — boost contrast here |

**`b_k`** is the brightness offset that keeps the local mean consistent after the `a`-weighted mapping.

In **Week 2**, the enhanced pixel will be computed as:

$$V'_i = a_i \cdot \text{CDF\_mapped}(V_i) + b_i$$

This blends the CDF-remapped value (from the histogram statistics module) with the GF coefficients to produce a contrast-enhanced image that **preserves edge sharpness**.

**OpenMP parallelisation:** The outer row loop carries a `#pragma omp parallel for schedule(dynamic)` directive. Each row is independent so there are no data races, and `schedule(dynamic)` balances load across cores automatically.

```
Input : H×W uint8 V channel
Output: (a_coeff[H×W], b_coeff[H×W])  — 2-D float64 arrays
```

---

### 2. Visualisations — `save_visualizations`

Produces two PNG files in the project root after every pipeline run.

#### `week1_gf_visual.png` — 2×3 figure

| Position | Panel | Description |
|---|---|---|
| Row 0, Col 0 | **Original image** | Colour (BGR→RGB) or grayscale depending on input |
| Row 0, Col 1 | **V channel** | Single-channel brightness map extracted from the input image |
| Row 0, Col 2 | **a-coefficient map** | Hot colourmap — bright = edge pixel, dark = flat pixel |
| Row 1, Col 0 | **b-coefficient map** | Viridis colourmap — brightness offset field |
| Row 1, Col 1 | **Edge highlight overlay** | Pixels where `a > 0.5` marked in red on the V channel |
| Row 1, Col 2 | **Colour vs V side-by-side** | Shows colour original next to its grayscale V channel |

#### `week1_distribution_visual.png` — 2×2 figure

| Position | Panel | Description |
|---|---|---|
| Row 0, Col 0 | **Full histogram** | 256-bin bar chart with mean `I_m` marked |
| Row 0, Col 1 | **Bi-histogram split** | Lower (blue) and upper (orange) sub-histograms |
| Row 1, Col 0 | **PDF + CDF** | PDF as bars, CDF as overlaid line (twin y-axis) |
| Row 1, Col 1 | **Original vs clipped CDFs** | Shows how APL clipping reshapes the CDF for each sub-histogram |

---

## How It Connects to the Rest of the Pipeline

```
V channel  (extracted from image)
    │
    ▼
compute_gf_coefficients  ──►  a_coeff[H×W],  b_coeff[H×W]
                                      │
                                      ▼
                               Week 2: V'_i = a_i × CDF_mapped(V_i) + b_i
                                      │
                                      ▼
                               Reconstruct colour image (H, S + V')

All pipeline data  ──►  save_visualizations  ──►  week1_gf_visual.png
                                              ──►  week1_distribution_visual.png
```

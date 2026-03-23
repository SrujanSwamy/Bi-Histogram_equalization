# Histogram Statistics & Adaptive Plateau Clipping

## Overview

This module builds the **full statistical model** of the image's intensity distribution and prepares the modified cumulative distribution functions (CDFs) that will drive histogram equalisation in Week 2. The key innovation here is the **Bi-Histogram** approach combined with an **entropy-based Adaptive Plateau Limit (APL)** — this prevents the classic over-enhancement and noise amplification that standard histogram equalisation produces.

All code is written in **C++17** with raw pointer arithmetic. Functions are exposed to Python via **pybind11**.

---

## File

```
src/member2/statistics.h
```

---

## What This Module Implements

### 1. Histogram — `compute_histogram`

Counts how many pixels fall into each of the 256 intensity bins (0–255):

$$h[k] = \text{number of pixels with } V = k, \quad k \in [0, 255]$$

The histogram is the starting point for all statistical analysis. Its shape reveals whether the image is dark, bright, low-contrast, etc.

```
Input : H×W uint8 V channel
Output: 256-element int array
```

---

### 2. Probability Density Function — `compute_pdf`

Normalises the histogram so all values sum to 1:

$$\text{PDF}[k] = \frac{h[k]}{N}, \quad N = H \times W$$

The PDF represents the probability that a randomly chosen pixel has intensity `k`. It is required to compute the CDF.

---

### 3. Cumulative Distribution Function — `compute_cdf`

Running sum of the PDF:

$$\text{CDF}[k] = \sum_{j=0}^{k} \text{PDF}[j]$$

The CDF maps any input intensity to its rank within the image. Standard histogram equalisation uses the CDF directly as a lookup table to remap pixels — stretching dark images, brightening flat regions.

---

### 4. Bi-Histogram Segmentation — `segment_histogram`

This is the core idea of the **Bi-HE** method. Instead of equalising the entire histogram at once (which shifts mean brightness), the histogram is **split at the mean intensity** `I_m`:

$$I_m = \text{round}\!\left(\frac{1}{N}\sum_i V_i\right)$$

- **Lower sub-histogram**: bins `[0, I_m]` — dark pixel population
- **Upper sub-histogram**: bins `[I_m+1, 255]` — bright pixel population

Each half is equalised **independently**. This constrains each sub-distribution to its own intensity range, so the mean brightness of the original image is preserved — a key advantage that standard HE does not have.

```
Input : H×W uint8 V channel
Output: (I_m, hist_lower[0..I_m], hist_upper[I_m+1..255])
```

---

### 5. Adaptive Plateau Limits & Clipping — `compute_apl_and_clip`

Standard HE amplifies noise in flat regions because even sparsely populated bins get fully stretched. The APL limits how tall any bin can be before it is clipped, preventing noise dominance.

The plateau limit is computed **separately for each sub-histogram** using its Shannon entropy:

$$T = \frac{N_{\text{sub}}}{n_{\text{bins}}} \quad \text{(mean bin frequency)}$$

$$E = -\sum_{i} p_i \log_2 p_i \quad \text{(discrete entropy of sub-histogram)}$$

$$E_{\max} = \log_2(n_{\text{bins}}) \quad \text{(maximum possible entropy)}$$

$$\text{PL} = \max\!\left(1,\ T \times \frac{E}{E_{\max}}\right)$$

**Intuition:**
- A flat, uniform sub-histogram has high entropy (E ≈ E_max) → PL is large → little clipping → full equalisation.
- A peaked, concentrated sub-histogram has low entropy → PL is small → heavy clipping → suppresses dominant bins → reduces over-enhancement.

After clipping, a new normalised CDF is built for each sub-histogram and returned as a single **256-element array** (lower CDF in indices `[0..I_m]`, upper CDF in `[I_m+1..255]`). This bi-CDF will be used in Week 2 to remap every pixel.

```
Input : hist_lower, hist_upper, I_m
Output: modified_cdf[256]  — ready for pixel remapping in Week 2
```

---

### Week 3: Temporal Smoothing

-   **Contribution**: Designed and implemented the `TemporalSmoother` class in C++ to stabilize video processing.
-   **Implementation**:
    -   A `TemporalSmoother` class was created in `src/core.cpp`.
    -   It uses three `std::deque<double>` buffers to store the most recent 5 values for the mean intensity (`I_m`), the low plateau limit (`PL_L`), and the high plateau limit (`PL_H`).
    -   The `update_and_get_smoothed()` method pushes new values to the deques, removes the oldest values if the size exceeds 5, and returns the simple moving average of the values in each buffer.
    -   The `apply_transformation` function was updated to accept a `double I_m` to work with the smoothed, continuous value.
-   **Impact**: This change is critical for video processing. It prevents drastic, frame-to-frame shifts in brightness and contrast that would otherwise cause a distracting "flickering" effect. By averaging these key parameters over a short time window, the enhancement evolves smoothly, leading to a much more visually consistent and pleasing result in the output video.

---

## How It Connects to the Rest of the Pipeline

```
V channel  (extracted from image)
    │
    ▼
compute_histogram
    │
    ├──► compute_pdf ──► compute_cdf   (for plotting / analysis)
    │
    ├──► segment_histogram             (split at mean I_m)
    │         │
    │         ▼
    └──► compute_apl_and_clip          (clip + build bi-CDF)
                │
                ▼
          modified_cdf[256]  ────────────────────────────────────►  Week 2
                                                                    (pixel remapping)
```

# Data Ingestion, Colour-Space Extraction & Evaluation Metrics

## Overview

This module is responsible for the very first stage of the pipeline: reading the raw image, converting it into a form the algorithm can work with, and measuring the quality of any image using five standard metrics. These metrics act as the **baseline** — they are computed on the original image in Week 1, and again on the enhanced image in Week 2 to objectively show how much the algorithm improved the picture.

All code is written in **C++17** using raw pointer arithmetic (no OpenCV in the backend). Functions are exposed to Python via **pybind11**.

---

## File

```
src/member1/data_metrics.h
```

---

## What This Module Implements

### 1. V-Channel Extraction — `extract_v_channel`

The paper operates on the **Value (brightness) channel** of an image, not the full colour image. This function performs that extraction.

- For a **grayscale** image (2-D array): returns a plain copy — the pixels already represent brightness.
- For a **colour** image (3-D, H×W×3): computes `V = max(R, G, B)` for every pixel.

Using `max(R, G, B)` instead of a weighted grey conversion preserves peak luminance faithfully, which is important for the histogram equalisation to work correctly. The Hue and Saturation channels are **not discarded** — they are stored separately in Python and will be recombined with the enhanced V channel in Week 2 to reconstruct a full colour output.

```
Input : H×W (grayscale) or H×W×3 (colour BGR)
Output: H×W single-channel uint8 brightness map
```

---

### 2. Mean Brightness — `compute_ambe`

Computes the **mean pixel intensity** of the V channel:

$$\text{AMBE} = \frac{1}{N} \sum_{i} V_i$$

In Week 2, the Absolute Mean Brightness Error is:

$$\text{AMBE} = | \bar{V}_{\text{original}} - \bar{V}_{\text{enhanced}} |$$

A lower AMBE means the algorithm preserved the natural brightness of the image rather than making it artificially too bright or too dark.

---

### 3. Standard Deviation — `compute_std`

Measures the **global contrast** of the image — how spread out the pixel intensities are:

$$\sigma = \sqrt{\frac{1}{N} \sum_{i} (V_i - \bar{V})^2}$$

A higher standard deviation after enhancement means the algorithm successfully stretched the contrast and made the image more visually distinct.

---

### 4. Contrast Improvement Index — `compute_cii`

Measures **local contrast** using a 3×3 sliding window around each pixel:

$$C_{\text{loc}} = \frac{V_{\max} - V_{\min}}{V_{\max} + V_{\min}}$$

The CII is the average of all local contrast values across the image. It captures fine texture and edge contrast that global standard deviation would miss.

---

### 5. Detail Enhancement — `compute_de`

Measures how well fine details (edges, textures) are preserved or enhanced. Computes the average absolute difference between each pixel and its 3×3 neighbourhood mean:

$$\text{DE} = \frac{1}{N} \sum_{i} |V_i - \mu_i^{\text{local}}|$$

A higher DE after enhancement means the algorithm sharpened fine structural details in the image.

---

### 6. Peak Signal Index — `compute_psi`

Counts the proportion of pixels that are at or near the maximum intensity (clipped highlights):

$$\text{PSI} = \frac{\text{pixels where } V \geq 250}{N}$$

A lower PSI is better — it means the algorithm did not over-brighten the image and clip highlight detail.

---

### Week 3: Continuous Metrics for Video

-   **Contribution**: Integrated the existing five mathematical metrics into the main Python video pipeline.
-   **Implementation**: The `pipeline.py` script now calls `compute_ambe`, `compute_std`, `compute_cii`, `compute_de`, and `compute_psi` periodically on the video stream (e.g., every 30 frames).
-   **Impact**: This provides continuous, real-time, quantitative feedback on the algorithm's performance during video processing. It allows us to monitor how temporal smoothing and other adaptive parameters affect the output quality from moment to moment, ensuring the enhancement remains stable and effective over time.

---

## How It Connects to the Rest of the Pipeline

```
Image file
    │
    ▼
extract_v_channel  ──────────────────────────────────────────────────────►  Histogram module
    │                                                                        (statistics)
    ▼
compute_ambe, compute_std, compute_cii, compute_de, compute_psi
    │
    ▼
Baseline metrics stored  ──── compared again in Week 2 after enhancement
```

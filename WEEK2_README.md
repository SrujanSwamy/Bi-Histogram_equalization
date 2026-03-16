# Week 2 Implementation: Adaptive Guided Filter & Transformation

## Overview
Week 2 implements the core adaptive enhancement logic by introducing integral images for fast local statistics, dynamic radius calculation based on image texture, and the final bi-histogram transformation using guided filter coefficients.

## Member Contributions

### **Member 1: Reconstruction**
Responsible for reassembling the processed components into the final output image.

*   **Step 1: HSV to RGB Merge (`merge_hsv_to_rgb`)**
    *   **Goal:** Recombine the image channels to produce the true-color output.
    *   **Implementation:** Merges the *original* Hue (H) and Saturation (S) channels (preserved from input) with the *enhanced* Value (V) channel. This ensures color integrity is maintained while brightness and contrast are enhanced.

### **Member 2: Adaptive Statistics & Dynamic Radius**
Responsible for enabling efficient local processing and determining optimal filter sizes.

*   **Step 2: Integral Images (`compute_integral_images`)**
    *   **Goal:** Enable $O(1)$ calculation of local sums for any window size.
    *   **Implementation:** Developed two "Summed Area Tables":
        1.  **Sum II:** Cumulative sum of pixel values.
        2.  **Squared Sum II:** Cumulative sum of squared pixel values.
    *   **Why:** Allows calculating the mean and variance of any rectangular region instantly, critical for the adaptive filter performance.

*   **Step 3: Busyness Map (`compute_busyness_map`)**
    *   **Goal:** Measure local texture complexity ("busyness") around each pixel.
    *   **Implementation:** Efficient calculation of local variance ($\sigma^2$) using the integral images:
        $$ Var(X) = E[X^2] - (E[X])^2 $$
    *   High variance indicates edges/texture; low variance indicates flat regions.

*   **Step 4: Dynamic Radius Mapping (`map_dynamic_radius`)**
    *   **Goal:** Assign an optimal filter radius for every pixel based on busyness.
    *   **Implementation:**
        *   **Flat regions (Low Busyness):** Radius = **7** (smoothing).
        *   **Textured regions (High Busyness):** Radius = **1** (detail preservation).
        *   **Transition:** Linearly interpolates between radii based on the busyness level.

### **Member 3: Adaptive Guided Filter & Transformation**
Responsible for the core enhancement logic using the derived statistical maps.

*   **Step 5: Adaptive Guided Filter (`adaptive_guided_filter`)**
    *   **Goal:** Calculate smoothing coefficients ($a, b$) using a variable window size.
    *   **Implementation:** Uses the **Dynamic Radius Map** for every pixel $(i, j)$. For each pixel, it looks up its specific radius $r_{i,j}$ and computes the local mean and variance using that window size via the integral images.

*   **Step 6: Normalized Edge Strength (`get_normalized_a`)**
    *   **Goal:** Create a standardized "edge map" from the filter coefficients.
    *   **Implementation:** Normalizes the $a$ coefficients (representing gradient strength) to the range $[0, 1]$. This map, $\hat{a}$, informs the final transformation about the presence and strength of edges.

*   **Step 7: Bi-Histogram Transformation (`apply_transformation`)**
    *   **Goal:** Apply the final pixel value adjustment.
    *   **Implementation:** Integrates all components into the final enhancement formula:
        $$ \lambda = (1 - \kappa \mu) + \kappa \cdot \hat{a} \quad \text{(if edge)} $$
        $$ V_{out} = V_{in} \cdot (1 - \lambda) + \text{Bi-HE}(V_{in}) \cdot \lambda $$
    *   **Logic:** Intelligently blends the **Original Pixel** (preserves natural look) with the **Equalized Pixel** (enhances contrast), controlled by the edge strength $\hat{a}$ and parameters $\kappa$ (strength) and $\mu$ (threshold).

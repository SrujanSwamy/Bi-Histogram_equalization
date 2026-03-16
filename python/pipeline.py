"""
Edge-Enhancing Bi-Histogram Equalisation — Week 1 Pipeline
===========================================================
Validates every C++ ↔ Python data bridge exposed by the he_core module.

Usage:
    python pipeline.py [path/to/image.jpg]

If no image path is provided, it defaults to ../test.jpg relative to this script.
"""

import os
import sys
import time

import cv2
import numpy as np

# Member 3 — visualization module (python/visualizations/visualize.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualizations.visualize import save_visualizations

# ---------------------------------------------------------------------------
# Locate the compiled he_core module (handles MSVC & single-config generators)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, ".."))

for _sub in ("build", "build/Release", "build/Debug",
             "build/RelWithDebInfo", "build/MinSizeRel"):
    _candidate = os.path.join(_project_root, _sub)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    import he_core
except ImportError as exc:
    print("ERROR: Could not import he_core. Did you compile the C++ module?")
    print(f"       ({exc})")
    print("\n  Build instructions:")
    print("    cd <project_root>")
    print('    mkdir build && cd build')
    print('    cmake .. -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"')
    print("    cmake --build . --config Release")
    sys.exit(1)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def load_image_uint8(img_path: str):
    """
    Load an image preserving its native channel count.
    - Grayscale TIFFs/PNGs/JPEGs are returned as (H, W)   uint8.
    - RGB/BGR images are returned as          (H, W, 3) uint8.
    16-bit images are scaled down to 8-bit automatically.
    Returns (img_uint8, bits_per_pixel_original).
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 0

    # Capture original bit depth before any conversion
    channels    = 1 if img.ndim == 2 else img.shape[2]
    bits_per_ch = img.dtype.itemsize * 8
    original_bpp = bits_per_ch * channels

    # Scale 16-bit → 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Drop alpha channel if present (H×W×4 → H×W×3)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return np.ascontiguousarray(img), original_bpp


def main() -> None:
    # --- Load image -------------------------------------------------------
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _project_root, "4.1.05.tiff")

    img, original_bpp = load_image_uint8(img_path)
    if img is None:
        print(f"ERROR: Could not load image '{img_path}'.")
        print("       or pass a path as argument:  python pipeline.py <img>")
        sys.exit(1)

    is_grayscale = (img.ndim == 2)
    channels     = 1 if is_grayscale else img.shape[2]
    mode_str     = "Grayscale" if is_grayscale else "RGB/BGR"

    print(f"Image loaded     : {img_path}")
    print(f"Shape / dtype    : {img.shape} / {img.dtype}")
    print(f"Detected mode    : {mode_str}")
    print(f"Original depth   : {original_bpp} bits/pixel  "
          f"({original_bpp // channels} bits/channel × {channels} channel{'s' if channels > 1 else ''})")
    if original_bpp > 24:
        print(f"  → High-bit image scaled to 8 bits/channel for processing")

    # ==== Step 1: Working Channel =========================================
    separator("Working Channel")

    if is_grayscale:
        # Grayscale: use the image directly — no V-extraction needed
        v_channel = img
        print(f"  Mode            : Grayscale — using pixel values directly")
        print(f"  Note            : V-channel extraction skipped (no colour info)")
    else:
        # RGB/BGR: extract HSV Value channel V = max(R, G, B) via C++
        # V channel is intentionally single-channel (grayscale) — it holds
        # only the brightness component.  Hue & Saturation are preserved
        # separately and will be recombined with the enhanced V in Week 2.
        v_channel = he_core.extract_v_channel(img)
        print(f"  Mode            : RGB — V channel extracted (V = max(R,G,B))")
        print(f"  V channel       : single-channel brightness map (grayscale by design)")
        print(f"  Colour info     : Hue & Saturation preserved — recombined in Week 2")

    print(f"  Working channel : shape {v_channel.shape}  dtype {v_channel.dtype}")
    print(f"  Intensity range : [{v_channel.min()}, {v_channel.max()}]")

    # ==== Step 2: Evaluation Metrics ======================================
    separator("Evaluation Metrics")

    ambe_val = he_core.compute_ambe(v_channel)
    std_val  = he_core.compute_std(v_channel)
    cii_val  = he_core.compute_cii(v_channel)
    de_val   = he_core.compute_de(v_channel)
    psi_val  = he_core.compute_psi(v_channel)

    print(f"  Mean Brightness (AMBE baseline) : {ambe_val:.4f}")
    print(f"  Standard Deviation              : {std_val:.4f}")
    print(f"  Contrast Improvement Index (CII): {cii_val:.6f}")
    print(f"  Discrete Entropy (DE)           : {de_val:.4f}")
    print(f"  Perceptual Sharpness Index (PSI): {psi_val:.4f}")

    # ==== Step 3: Histogram / PDF / CDF ===================================
    separator("Histogram / PDF / CDF")

    hist = he_core.compute_histogram(v_channel)
    pdf  = he_core.compute_pdf(hist)
    cdf  = he_core.compute_cdf(pdf)

    # Non-zero bins = number of distinct intensity levels actually used
    nonzero_bins = int(np.count_nonzero(hist))
    peak_bin     = int(np.argmax(hist))
    peak_count   = int(hist[peak_bin])

    print(f"  Histogram sum   : {hist.sum()}  (= total pixels, always H×W — correct)")
    print(f"  Non-zero bins   : {nonzero_bins} / 256  (distinct intensity levels)")
    print(f"  Peak bin        : intensity {peak_bin}  with {peak_count} pixels")
    print(f"  PDF sum         : {pdf.sum():.6f}")
    print(f"  CDF[255]        : {cdf[255]:.6f}")

    assert hist.sum() == v_channel.size, "Histogram pixel count mismatch!"
    assert abs(pdf.sum() - 1.0) < 1e-6,  "PDF does not sum to 1!"
    assert abs(cdf[255] - 1.0) < 1e-6,   "CDF does not reach 1!"
    print("  Assertions      : ALL PASSED")

    # ==== Bi-Histogram Segmentation =======================================
    separator("Bi-Histogram Segmentation")

    I_m, hist_lower, hist_upper = he_core.segment_histogram(v_channel)
    print(f"  Mean Intensity (I_m) : {I_m}")
    print(f"  Lower sub-hist bins  : {len(hist_lower)}  [0, {I_m}]")
    print(f"  Upper sub-hist bins  : {len(hist_upper)}  [{I_m + 1}, 255]")
    print(f"  Lower pixel count    : {hist_lower.sum()}")
    print(f"  Upper pixel count    : {hist_upper.sum()}")
    print(f"  Total                : {hist_lower.sum() + hist_upper.sum()}"
          f"  (expected {v_channel.size})")

    assert hist_lower.sum() + hist_upper.sum() == v_channel.size, \
        "Sub-histogram counts do not add up!"
    print("  Assertions           : ALL PASSED")

    # ==== APL Computation & Histogram Clipping ============================
    separator("Adaptive Plateau Limits & Clipping")

    modified_cdf = he_core.compute_apl_and_clip(hist, I_m)
    print(f"  Modified CDF shape : {modified_cdf.shape}")
    print(f"  CDF[0]             : {modified_cdf[0]:.6f}")
    print(f"  CDF[I_m={I_m}]       : {modified_cdf[I_m]:.6f}")
    if I_m < 255:
        print(f"  CDF[I_m+1={I_m + 1}]     : {modified_cdf[I_m + 1]:.6f}")
    print(f"  CDF[255]           : {modified_cdf[255]:.6f}")

    # The lower and upper CDFs should each reach ≈1.0 at their last bin
    assert abs(modified_cdf[I_m] - 1.0) < 1e-6, \
        f"Lower CDF does not reach 1.0 (got {modified_cdf[I_m]})"
    assert abs(modified_cdf[255] - 1.0) < 1e-6, \
        f"Upper CDF does not reach 1.0 (got {modified_cdf[255]})"
    print("  Assertions         : ALL PASSED")

    # ==== Guided Filter Coefficients (OpenMP) =============================
    separator("Guided Filter Coefficients")

    omp_threads = he_core.get_omp_max_threads()
    print(f"  OpenMP max threads : {omp_threads}")

    t0 = time.perf_counter()
    a_coeff, b_coeff = he_core.compute_gf_coefficients(v_channel)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  a_coeff shape      : {a_coeff.shape}  "
          f"range [{a_coeff.min():.6f}, {a_coeff.max():.6f}]")
    print(f"  b_coeff shape      : {b_coeff.shape}  "
          f"range [{b_coeff.min():.4f}, {b_coeff.max():.4f}]")
    print(f"  Computation time   : {elapsed_ms:.2f} ms")

    assert a_coeff.shape == v_channel.shape, "a_coeff shape mismatch!"
    assert b_coeff.shape == v_channel.shape, "b_coeff shape mismatch!"
    assert b_coeff.shape == v_channel.shape, "b_coeff shape mismatch!"
    print("  Assertions         : ALL PASSED")

    # ==== Visualizations (Member 3) =======================================
    separator("Visualizations ")

    print(f"  Generating Week 1 visualizations in: {_project_root}")
    save_visualizations(img, v_channel, a_coeff, b_coeff,
                        hist, pdf, cdf, modified_cdf,
                        I_m, hist_lower, hist_upper,
                        is_grayscale, _project_root)
    print("  [SUCCESS] Visualizations saved.")

    # ==== Week 2 - Integral Images & Dynamic Radius =======================
    separator("Integral Images & Dynamic Radius ")

    # Compute Integral Images (Sum and Squared Sum)
    sum_ii, sq_sum_ii = he_core.compute_integral_images(v_channel)
    print(f"  Integral Image shape : {sum_ii.shape}")

    # Compute Busyness Map (r=1 for 3x3 local variance)
    # C++ Signature: compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    rows, cols = v_channel.shape
    busyness_map = he_core.compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    print(f"  Busyness Map         : range [{busyness_map.min():.4f}, {busyness_map.max():.4f}]")

    # Map Busyness to Dynamic Radius
    # Flat (low busyness) -> Radius 7
    # Edge (high busyness) -> Radius 1
    # Interpolated linearly.
    radius_map = he_core.map_dynamic_radius(busyness_map, 1, 7)
    print(f"  Dynamic Radius Map   : range [{radius_map.min()}, {radius_map.max()}] (type {radius_map.dtype})")

    # ==== Adapter Guided Filter & Transformation ==========================
    separator("Adaptive Guided Filter & Transformation ")

    # Compute Adaptive Coefficients (a, b)
    # Using the variable radius map and integral images for O(1) box sums
    a_adapt, b_adapt = he_core.adaptive_guided_filter(v_channel, sum_ii, sq_sum_ii, radius_map)
    print(f"  Adaptive 'a' coeff   : range [{a_adapt.min():.4f}, {a_adapt.max():.4f}]")

    # Normalize 'a' to get edge coefficients (a_hat)
    a_hat = he_core.get_normalized_a(a_adapt)
    print(f"  Normalized 'a_hat'   : range [{a_hat.min():.4f}, {a_hat.max():.4f}]")

    # Apply Final Transformation
    # Uses: V channel, a_hat, modified CDF (from Step 5), and parameters mu, kappa.
    # mu: threshold for flat vs edge. a_hat < mu is flat.
    # kappa: enhancement strength.
    mu_param = 0.5
    kappa_param = 0.6
    v_enhanced = he_core.apply_transformation(v_channel, a_hat, modified_cdf, mu_param, kappa_param, I_m)
    print(f"  Enhanced V shape     : {v_enhanced.shape}")
    print(f"  Transformation Params: mu={mu_param}, kappa={kappa_param}")

    # ==== Reconstruction & Display ========================================
    separator("Reconstruction & Display ")

    if is_grayscale:
        final_img = v_enhanced
        print("  Grayscale input -> returning enhanced V channel directly.")
    else:
        # Separate original H, S using OpenCV
        # Load image as BGR (OpenCV default)
        # Convert to HSV
        # Pass H, S, V_enhanced to C++ merge function
        hsv_orig = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv_orig)
        
        print("  Merging Original H, S with Enhanced V...")
        final_rgb = he_core.merge_hsv_to_rgb(h, s, v_enhanced)
        
        # Convert back to BGR for OpenCV consistency / display
        final_img = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        print("  Reconstruction complete.")

    # Show side-by-side
    if final_img.ndim == 2:
        final_disp = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
    else:
        final_disp = final_img

    original_disp = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack((original_disp, final_disp))
    
    # Save validation image
    out_file = os.path.join(_project_root, "week2_result.png")
    cv2.imwrite(out_file, combined)
    print(f"\n  [SUCCESS] Result saved to: {out_file}")
    
    # Try to display (may fail in headless envs)
    try:
        cv2.imshow("Original (Left) vs Enhanced (Right)", combined)
        print("  Press any key in the window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("  (Skipping window display - minimal environment detected)")

    # ==== Summary =========================================================
    separator("Week 2 Implementation Complete")
    print()


if __name__ == "__main__":
    main()

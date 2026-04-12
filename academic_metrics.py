"""
academic_metrics.py
===================
Mathematical comparison script for final report deliverables.

Compares three methods on one test image:
1) Original
2) OpenCV Histogram Equalization (on V channel)
3) Custom he_core enhancement pipeline

Metrics are computed via C++ backend functions:
- AMBE
- STD
- CII
- DE
- PSI

Usage:
    python academic_metrics.py --image 5.1.11.jpg
"""

import argparse
import os
import sys
from typing import Dict

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Module loading helper
# -----------------------------------------------------------------------------
def load_he_core() -> object:
    """Load `he_core` from CMake build outputs with Windows DLL support."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    for sub in ("build", "build/Release", "build/Debug", "build/RelWithDebInfo", "build/MinSizeRel"):
        candidate = os.path.join(project_root, sub)
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    if sys.platform == "win32":
        for dll_dir in (
            r"C:\msys64\ucrt64\bin",
            r"C:\msys64\mingw64\bin",
        ):
            if os.path.isdir(dll_dir):
                os.add_dll_directory(dll_dir)

    import he_core  # pylint: disable=import-error, import-outside-toplevel

    return he_core


def run_custom_pipeline(he_core: object, bgr_img: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the full custom enhancement and return intermediate/final outputs."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hist = he_core.compute_histogram(v)
    i_m = float(np.mean(v))
    modified_cdf = he_core.compute_apl_and_clip(hist, int(i_m))

    rows, cols = v.shape
    sum_ii, sq_sum_ii = he_core.compute_integral_images(v)
    busyness = he_core.compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    radius = he_core.map_dynamic_radius(busyness, 2, 16)

    a, _b = he_core.adaptive_guided_filter(v, sum_ii, sq_sum_ii, radius)
    a_hat = he_core.get_normalized_a(a)
    v_out = he_core.apply_transformation(v, a_hat, modified_cdf, 0.5, 1.5, i_m)

    rgb_out = he_core.merge_hsv_to_rgb(h, s, v_out)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    return {
        "v_original": v,
        "v_custom": v_out,
        "bgr_custom": bgr_out,
    }


def compute_all_metrics(he_core: object, v_channel: np.ndarray) -> Dict[str, float]:
    """Compute the five requested report metrics for a V-channel image."""
    return {
        "AMBE": float(he_core.compute_ambe(v_channel)),
        "STD": float(he_core.compute_std(v_channel)),
        "CII": float(he_core.compute_cii(v_channel)),
        "DE": float(he_core.compute_de(v_channel)),
        "PSI": float(he_core.compute_psi(v_channel)),
    }


def print_markdown_table(original: Dict[str, float], opencv_he: Dict[str, float], custom: Dict[str, float]) -> None:
    """Print a Markdown table suitable for direct academic report insertion."""
    headers = ["Method", "AMBE", "STD", "CII", "DE", "PSI"]
    print("\n| " + " | ".join(headers) + " |")
    print("|---|---:|---:|---:|---:|---:|")

    def row(name: str, m: Dict[str, float]) -> None:
        print(
            f"| {name} | "
            f"{m['AMBE']:.4f} | {m['STD']:.4f} | {m['CII']:.6f} | {m['DE']:.4f} | {m['PSI']:.4f} |"
        )

    row("Original", original)
    row("OpenCV HE", opencv_he)
    row("Custom Bi-HE + GIF", custom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Academic metric comparison (Original vs OpenCV HE vs Custom).")
    parser.add_argument("--image", default="5.1.11.jpg", help="Path to input test image.")
    parser.add_argument("--save", action="store_true", help="Save generated method outputs for report appendix.")
    args = parser.parse_args()

    he_core = load_he_core()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    # Original V channel
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v_original = cv2.split(hsv)

    # OpenCV baseline: equalize V only, then reconstruct image.
    v_he = cv2.equalizeHist(v_original)
    hsv_he = cv2.merge([h, s, v_he])
    bgr_he = cv2.cvtColor(hsv_he, cv2.COLOR_HSV2BGR)

    # Custom method
    custom_out = run_custom_pipeline(he_core, bgr)

    # Compute metrics for all three methods.
    metrics_original = compute_all_metrics(he_core, v_original)
    metrics_opencv = compute_all_metrics(he_core, v_he)
    metrics_custom = compute_all_metrics(he_core, custom_out["v_custom"])

    print_markdown_table(metrics_original, metrics_opencv, metrics_custom)

    if args.save:
        cv2.imwrite("academic_original.png", bgr)
        cv2.imwrite("academic_opencv_he.png", bgr_he)
        cv2.imwrite("academic_custom.png", custom_out["bgr_custom"])
        print("\nSaved: academic_original.png, academic_opencv_he.png, academic_custom.png")


if __name__ == "__main__":
    main()

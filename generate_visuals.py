"""
generate_visuals.py
===================
Presentation-graphics generator for final academic deliverables.

Creates three high-resolution figures:
1) Bi-Histogram plot (with mean split and plateau lines)
2) Busyness map heatmap
3) Intensity-frequency comparison (input vs enhanced)

Usage:
    python generate_visuals.py --image 5.1.11.jpg
"""

import argparse
import os
import sys
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Module loading helper
# -----------------------------------------------------------------------------
def load_he_core() -> object:
    """Load `he_core` from build artifacts with Windows DLL support."""
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


# -----------------------------------------------------------------------------
# Plateau-limit reconstruction in Python (for plotting)
# Mirrors the C++ entropy-driven clipping logic.
# -----------------------------------------------------------------------------
def compute_plateau_limits_python(hist: np.ndarray, i_m: int) -> Tuple[float, float]:
    """Estimate PL_L and PL_H from histogram and split intensity i_m."""
    i_m = int(np.clip(i_m, 0, 254))
    hist = hist.astype(np.float64)

    def sub_entropy(lo: int, hi: int) -> float:
        sub = hist[lo : hi + 1]
        n_sub = np.sum(sub)
        if n_sub <= 0:
            return 0.0
        p = sub[sub > 0] / n_sub
        return float(-np.sum(p * np.log2(p)))

    # Lower partition [0, i_m]
    n_l = i_m + 1
    n_pixels_l = float(np.sum(hist[: i_m + 1]))
    t_l = (n_pixels_l / n_l) if (n_l > 0 and n_pixels_l > 0) else 0.0
    e_l = sub_entropy(0, i_m)
    emax_l = np.log2(n_l) if n_l > 1 else 1.0
    pl_l = max(1.0, t_l * e_l / (emax_l + 1e-10))

    # Upper partition [i_m+1, 255]
    n_h = 255 - i_m
    n_pixels_h = float(np.sum(hist[i_m + 1 :]))
    t_h = (n_pixels_h / n_h) if (n_h > 0 and n_pixels_h > 0) else 0.0
    e_h = sub_entropy(i_m + 1, 255)
    emax_h = np.log2(n_h) if n_h > 1 else 1.0
    pl_h = max(1.0, t_h * e_h / (emax_h + 1e-10))

    return pl_l, pl_h


def get_output_dir(image_path: str) -> str:
    """Return outputs/<image_name>/ and create it if missing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
    if not base_name:
        base_name = "unknown"
    out_dir = os.path.join(script_dir, "outputs", base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report visuals from the custom C++ pipeline.")
    parser.add_argument("--image", default="5.1.11.jpg", help="Path to input test image.")
    parser.add_argument(
        "--kappa",
        type=float,
        default=5.0,
        help="Sharpening parameter kappa (default: 5.0, aligned with pipeline.py).",
    )
    args = parser.parse_args()

    he_core = load_he_core()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    out_dir = get_output_dir(args.image)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # -------------------------------------------------------------------------
    # Extract all key intermediate arrays from the custom pipeline.
    # -------------------------------------------------------------------------
    hist_cpp = he_core.compute_histogram(v)
    hist = np.asarray(hist_cpp).reshape(-1)

    # Keep the visual pipeline aligned with python/pipeline.py:
    # Im is floor(mean(V)) and cdf split uses Im clamped to [0, 254].
    i_m = float(int(np.mean(v)))
    i_m_int = int(np.clip(i_m, 0, 254))

    pl_l, pl_h = compute_plateau_limits_python(hist, i_m_int)

    modified_cdf = he_core.compute_apl_and_clip(hist_cpp, i_m_int)
    rows, cols = v.shape
    sum_ii, sq_sum_ii = he_core.compute_integral_images(v)
    busyness = he_core.compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    radius = he_core.map_dynamic_radius(busyness, 2, 6)
    a, _b = he_core.adaptive_guided_filter(v, sum_ii, sq_sum_ii, radius)
    a_hat = he_core.get_normalized_a(a)
    mu = he_core.compute_mu_from_a_hat(a_hat)
    v_out = he_core.apply_transformation(v, a_hat, modified_cdf, mu, args.kappa, float(i_m))

    # Optional output image for appendix.
    rgb_out = he_core.merge_hsv_to_rgb(h, s, v_out)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "enhanced_for_visual_report.png"), bgr_out)

    # -------------------------------------------------------------------------
    # Plot 1: Bi-Histogram with split and plateau lines.
    # -------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 7))
    ax1 = fig1.add_subplot(111)

    x = np.arange(256)
    ax1.plot(x, hist, color="royalblue", linewidth=1.6, label="Original Histogram")

    ax1.axvline(i_m, color="darkred", linestyle="--", linewidth=2.0, label=f"Mean Intensity I_m = {i_m:.0f}")

    # Plateau limits are drawn over their corresponding partitions.
    ax1.hlines(pl_l, 0, i_m, colors="seagreen", linestyles="-.", linewidth=2.0, label=f"PL_L = {pl_l:.2f}")
    ax1.hlines(pl_h, i_m, 255, colors="orange", linestyles="-.", linewidth=2.0, label=f"PL_H = {pl_h:.2f}")

    ax1.set_title("Bi-Histogram Segmentation and Plateau Clipping", fontsize=14, pad=12)
    ax1.set_xlabel("Intensity Level", fontsize=12)
    ax1.set_ylabel("Pixel Count", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "bi_histogram_visual.png"), dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Busyness Map heatmap.
    # -------------------------------------------------------------------------
    busyness_np = np.asarray(busyness)

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(busyness_np, cmap="magma")

    ax2.set_title("Busyness Map Heatmap (Texture/Edge Sensitivity)", fontsize=14, pad=12)
    ax2.set_xlabel("X (columns)", fontsize=12)
    ax2.set_ylabel("Y (rows)", fontsize=12)

    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("Normalized Local Variance", rotation=90)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "busyness_map_visual.png"), dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Plot 3: Intensity-frequency bar plot (input vs enhanced)
    # -------------------------------------------------------------------------
    hist_in = cv2.calcHist([v], [0], None, [256], [0, 256]).flatten()
    hist_out = cv2.calcHist([v_out], [0], None, [256], [0, 256]).flatten()

    fig3 = plt.figure(figsize=(12, 7))
    ax3 = fig3.add_subplot(111)
    x = np.arange(256)

    ax3.bar(x, hist_in, width=1.0, color="steelblue", alpha=0.55, label="Input Intensity Frequency")
    ax3.bar(x, hist_out, width=1.0, color="crimson", alpha=0.45, label="Enhanced Intensity Frequency")

    ax3.set_title("Intensity vs Frequency (Bar Plot): Input vs Enhanced", fontsize=14, pad=12)
    ax3.set_xlabel("Intensity", fontsize=12)
    ax3.set_ylabel("Frequency (Pixel Count)", fontsize=12)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper right")

    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "intensity_frequency_comparison.png"), dpi=300)
    plt.close(fig3)

    print(f"Saved high-resolution figures to: {out_dir}")
    print("- bi_histogram_visual.png")
    print("- busyness_map_visual.png")
    print("- intensity_frequency_comparison.png")
    print("- enhanced_for_visual_report.png")


if __name__ == "__main__":
    main()

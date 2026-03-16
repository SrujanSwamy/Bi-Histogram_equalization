"""
Member 3 — Guided Filter Visualizations
========================================
Saves two figures to the project output directory:

  week1_gf_visual.png           — Original colour + GF spatial coefficient maps
  week1_distribution_visual.png — Pixel intensity distribution plots

Call:
    from member3.visualize import save_visualizations
    save_visualizations(img, v_channel, a_coeff, b_coeff,
                        hist, pdf, cdf, modified_cdf,
                        I_m, hist_lower, hist_upper,
                        is_grayscale, output_dir)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — writes PNG, no display needed
import matplotlib.pyplot as plt


def save_visualizations(img, v_channel, a_coeff, b_coeff,
                        hist, pdf, cdf, modified_cdf,
                        I_m, hist_lower, hist_upper,
                        is_grayscale: bool = False,
                        output_dir: str = ".") -> None:
    """Generate and save all Member-3 visualizations."""

    x_bins = np.arange(256)

    # =========================================================================
    #  FIGURE 1 — Original image + Guided Filter spatial maps (2×3)
    # =========================================================================
    edge_mask = (a_coeff > 0.5).astype(np.uint8) * 255
    edge_pct  = edge_mask.sum() / 255 / v_channel.size * 100

    overlay = np.stack([v_channel, v_channel, v_channel], axis=-1)
    overlay[edge_mask == 255] = [220, 50, 50]   # red = edge pixel

    fig1, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig1.suptitle("Member 3 — Guided Filter: Pixel-Level Analysis", fontsize=13)

    # Panel 1 — Original image (colour if RGB, gray if grayscale)
    ax = axes[0, 0]
    if is_grayscale:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Original Image\n(Grayscale)")
    else:
        # OpenCV loads BGR — convert to RGB for correct colour display
        ax.imshow(img[:, :, ::-1])
        ax.set_title("Original Image\n(Colour — BGR→RGB)")
    ax.axis("off")

    # Panel 2 — V channel (always grayscale by design)
    ax = axes[0, 1]
    im = ax.imshow(v_channel, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"V Channel (brightness only)\n"
        f"range [{v_channel.min()}, {v_channel.max()}]\n"
        f"← grayscale by design (luminance only)"
    )
    ax.axis("off")
    fig1.colorbar(im, ax=ax, fraction=0.046)

    # Panel 3 — a-coefficient map (edge gain)
    ax = axes[0, 2]
    im = ax.imshow(a_coeff, cmap="hot", vmin=0, vmax=1)
    ax.set_title(
        f"a-coefficient (edge gain)\n"
        f"range [{a_coeff.min():.3f}, {a_coeff.max():.3f}]\n"
        f"bright = edge  |  dark = flat"
    )
    ax.axis("off")
    fig1.colorbar(im, ax=ax, fraction=0.046)

    # Panel 4 — b-coefficient map (brightness offset)
    ax = axes[1, 0]
    im = ax.imshow(b_coeff, cmap="viridis")
    ax.set_title(
        f"b-coefficient (brightness offset)\n"
        f"range [{b_coeff.min():.1f}, {b_coeff.max():.1f}]\n"
        f"b = μ · (1 − a)"
    )
    ax.axis("off")
    fig1.colorbar(im, ax=ax, fraction=0.046)

    # Panel 5 — Edge pixels highlighted in red
    ax = axes[1, 1]
    ax.imshow(overlay)
    ax.set_title(
        f"Edge pixels (a > 0.5) highlighted\n"
        f"{edge_pct:.1f}% of pixels are edge / texture"
    )
    ax.axis("off")

    # Panel 6 — Side-by-side colour vs V channel (only for RGB images)
    ax = axes[1, 2]
    if not is_grayscale:
        # Horizontal stack: colour | V channel (as gray)
        colour_rgb  = img[:, :, ::-1]                          # BGR→RGB
        v_rgb       = np.stack([v_channel] * 3, axis=-1)       # gray→RGB
        divider     = np.ones((img.shape[0], 4, 3), dtype=np.uint8) * 200
        combined    = np.hstack([colour_rgb, divider, v_rgb])
        ax.imshow(combined)
        ax.set_title("Colour (left) vs V channel (right)\nV discards Hue & Saturation intentionally")
    else:
        ax.imshow(v_channel, cmap="gray", vmin=0, vmax=255)
        ax.set_title("V channel\n(same as input — grayscale image)")
    ax.axis("off")

    fig1.tight_layout()
    out1 = os.path.join(output_dir, "week1_gf_visual.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [Member 3] Saved GF map          : {out1}")

    # =========================================================================
    #  FIGURE 2 — Pixel Intensity Distribution (2×2)
    # =========================================================================
    hist_vals  = hist.astype(np.float64)

    lower_full = np.zeros(256)
    upper_full = np.zeros(256)
    lower_full[:I_m + 1]    = hist_lower
    upper_full[I_m + 1:256] = hist_upper

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("Member 3 — Pixel Intensity Distribution Analysis", fontsize=13)

    # Panel 1 — Full histogram
    ax = axes2[0, 0]
    ax.bar(x_bins, hist_vals, width=1.0, color="#3a7ebf", edgecolor="none")
    ax.axvline(I_m, color="red", linewidth=1.5, linestyle="--",
               label=f"Mean I_m = {I_m}")
    ax.set_title("Pixel Intensity Histogram")
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim(0, 255)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2 — Bi-histogram split
    ax = axes2[0, 1]
    ax.bar(x_bins[:I_m + 1], lower_full[:I_m + 1],
           width=1.0, color="#3a7ebf", edgecolor="none",
           label=f"Lower [0, {I_m}]")
    ax.bar(x_bins[I_m + 1:], upper_full[I_m + 1:],
           width=1.0, color="#f07020", edgecolor="none",
           label=f"Upper [{I_m + 1}, 255]")
    ax.axvline(I_m, color="red", linewidth=1.5, linestyle="--")
    ax.set_title("Bi-Histogram Split at Mean Intensity")
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim(0, 255)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3 — PDF bars + CDF line
    ax  = axes2[1, 0]
    ax2r = ax.twinx()
    ax.bar(x_bins, pdf, width=1.0, color="#3a7ebf",
           edgecolor="none", alpha=0.6, label="PDF")
    ax2r.plot(x_bins, cdf, color="#e03030", linewidth=2, label="CDF")
    ax.set_title("PDF (bars) and CDF (line)")
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Probability (PDF)", color="#3a7ebf")
    ax2r.set_ylabel("Cumulative Probability (CDF)", color="#e03030")
    ax.set_xlim(0, 255)
    ax2r.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

    # Panel 4 — Original vs APL-clipped CDFs
    ax = axes2[1, 1]
    ax.plot(x_bins, cdf, color="#3a7ebf", linewidth=2,
            label="Original CDF")
    ax.plot(x_bins[:I_m + 1], modified_cdf[:I_m + 1],
            color="#f07020", linewidth=2, linestyle="--",
            label=f"Clipped Lower [0, {I_m}]")
    ax.plot(x_bins[I_m + 1:], modified_cdf[I_m + 1:],
            color="#20a020", linewidth=2, linestyle="--",
            label=f"Clipped Upper [{I_m + 1}, 255]")
    ax.axvline(I_m, color="red", linewidth=1.2, linestyle=":")
    ax.set_title("Original vs APL-Clipped CDF")
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.4)

    fig2.tight_layout()
    out2 = os.path.join(output_dir, "week1_distribution_visual.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [Member 3] Saved distribution    : {out2}")
    print(f"  [Member 3] Edge pixels (a > 0.5) : {edge_pct:.1f}% of image")

"""
Edge-Enhancing Bi-Histogram Equalisation — Week 3 Unified Pipeline
===================================================================
Handles both Static Images & Live Video Feeds.
Includes continuous Mathematical Output (AMBE, Std, CII, DE, PSI).
Implements Temporal Smoothing for video consistency.
Uses Multi-threaded C++ Backend (OpenMP).

Usage:
    python pipeline.py [path/to/image_or_video] (or '0' for webcam)

KEY FIXES (vs original):
  1. mu is now computed per-frame as mean(a_hat) — paper Eq.(21) — instead
     of being a fixed CLI parameter. The '--mu' flag has been removed.
  2. APL formula in statistics.h corrected to paper Eqs.(12-13):
         PL_s = (H_sub / H_input) * Avg_sub
     replacing the incorrect T * E / Emax formula.
  3. I_m uses floor() as per paper Eq.(10), not round().
  4. apply_transformation uses std::round() for correct integer mapping.
  5. kappa default changed to 5.0 (paper's experimental setting, Section 4).
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# DLL Hot-Fix for Windows/MinGW
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    mingw_bin_path = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(mingw_bin_path):
        print(f"System: Adding MinGW binary path to DLL search path: {mingw_bin_path}")
        os.add_dll_directory(mingw_bin_path)
    else:
        print(f"Warning: MinGW binary path not found at '{mingw_bin_path}'.")
        print("         The 'he_core' module may fail to load if its dependencies are not found.")

# ---------------------------------------------------------------------------
# Locate the compiled he_core module
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, ".."))

for _sub in ("build", "build/Release", "build/Debug", "build/RelWithDebInfo", "build/MinSizeRel"):
    _candidate = os.path.join(_project_root, _sub)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    import he_core
except ImportError as exc:
    print(f"ERROR: Could not import he_core. ({exc})")
    print("Ensure the C++ module is compiled successfully.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Math Evaluation Helper
# ---------------------------------------------------------------------------
def print_math_metrics(v_original, v_enhanced, label="Image"):
    """
    Calculates and prints the 5 strict metrics from the research paper.
    """
    print(f"\n--- Mathematical Output ({label}) ---")

    # Original Stats
    print(" Original V-Channel:")
    print(f"   Mean Brightness : {he_core.compute_ambe(v_original):.4f}")
    print(f"   Std Deviation   : {he_core.compute_std(v_original):.4f}")

    # Enhanced Stats
    ambe_val = he_core.compute_ambe(v_enhanced)
    std_val  = he_core.compute_std(v_enhanced)
    cii_val  = he_core.compute_cii(v_enhanced)
    de_val   = he_core.compute_de(v_enhanced)
    psi_val  = he_core.compute_psi(v_enhanced)

    print(" Enhanced V-Channel:")
    print(f"   Mean Brightness : {ambe_val:.4f}")
    print(f"   Std Deviation   : {std_val:.4f}")
    print(f"   Contrast (CII)  : {cii_val:.6f}")
    print(f"   Entropy (DE)    : {de_val:.4f}")
    print(f"   Sharpness (PSI) : {psi_val:.4f}")
    print("-------------------------------------")


# ---------------------------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------------------------
def process_frame(frame: np.ndarray, smoother=None, kappa: float = 5.0):
    """
    Unified processing function for both image and video modes.

    Parameters
    ----------
    frame   : BGR image (numpy uint8 H×W×3)
    smoother: optional TemporalSmoother instance (video mode only)
    kappa   : sharpening parameter κ — paper uses 5 in experiments (Section 4)

    Returns
    -------
    bgr_out  : enhanced BGR image
    v        : original V channel
    v_out    : enhanced V channel

    Pipeline (matches paper Fig. 1b exactly):
      1. BGR → HSV; extract V channel
      2. Compute I_m = floor(mean(V))  [paper Eq. 10]
      3. Optional temporal smoothing of I_m (video mode)
      4. Compute 256-bin histogram of V
      5. Compute adaptive plateau limits & clip; build modified CDF
         [paper Eqs. 12-18]
      6. Compute integral images → busyness map → dynamic radius map
      7. Adaptive guided filter → a, b coefficients  [paper Eq. 7-9]
      8. Normalise a → â  [paper: â = a / max(a)]
      9. Compute μ = mean(â) over entire image  [paper Eq. 21] ← FIX
     10. Apply bi-histogram transformation with λ from Eq. 21  [paper Eq. 19]
     11. Merge enhanced V back with H, S → RGB → BGR
    """
    # 1. Convert to HSV and split
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. I_m = floor(mean(V))  — paper Eq.(10) uses floor, not round
    I_m = float(int(np.mean(v)))   # int() truncates = floor for non-negative

    # 3. Temporal smoothing of I_m (video only)
    if smoother is not None:
        I_m, _, _ = smoother.update_and_get_smoothed(I_m, 0.0, 0.0)

    I_m_int = int(np.clip(I_m, 0, 254))

    # 4. Full 256-bin histogram
    hist = he_core.compute_histogram(v)

    # 5. Adaptive plateau limits + clip + build modified bi-CDF
    #    statistics.h now implements paper Eqs.(12-18) correctly.
    modified_cdf = he_core.compute_apl_and_clip(hist, I_m_int)

    # 6. Integral images → busyness → dynamic radius
    rows, cols = v.shape
    sum_ii, sq_sum_ii = he_core.compute_integral_images(v)
    busyness = he_core.compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    # r_max kept at 6 to avoid halo bleeding at hard edges
    radius = he_core.map_dynamic_radius(busyness, 2, 6)

    # 7. Adaptive guided filter (self-guided: guidance = input = V)
    a, b_gf = he_core.adaptive_guided_filter(v, sum_ii, sq_sum_ii, radius)

    # 8. Normalise a → â   [â = a / max(a)]
    a_hat = he_core.get_normalized_a(a)

    # 9. μ = mean(â) — computed per-frame from the actual a_hat array.
    #    Paper Eq.(21): λ uses μ as the image-level average of â.
    #    The old code passed a fixed CLI '--mu 0.5', which is WRONG —
    #    μ varies per image and must be derived from the GF output.
    mu = he_core.compute_mu_from_a_hat(a_hat)

    # 10. Apply bi-histogram transformation
    v_out = he_core.apply_transformation(v, a_hat, modified_cdf, mu, kappa, float(I_m))

    # 11. Merge & return
    rgb_out = he_core.merge_hsv_to_rgb(h, s, v_out)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    return bgr_out, v, v_out


def build_diff_view(original_bgr: np.ndarray, enhanced_bgr: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """
    Build a colorized absolute-difference heatmap to highlight subtle changes.
    """
    diff = cv2.absdiff(original_bgr, enhanced_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_boosted = np.clip(diff_gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff_boosted, cv2.COLORMAP_TURBO)


def get_output_dir(input_source: str) -> str:
    """
    Return output folder as outputs/<input_name>/ and create it if missing.
    """
    if str(input_source) == "0":
        base_name = "webcam"
    else:
        base_name = os.path.splitext(os.path.basename(str(input_source)))[0]
        if not base_name:
            base_name = "unknown"

    out_dir = os.path.join(_project_root, "outputs", base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_image_mode(path, kappa: float, show_diff: bool, diff_gain: float):
    print(f"Mode: Image | Path: {path}")
    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load image.")
        return

    start_time = time.time()
    result, v_orig, v_enh = process_frame(img, smoother=None, kappa=kappa)
    elapsed = (time.time() - start_time) * 1000.0

    print(f"Processing Complete: {elapsed:.2f} ms")

    # Print the exact math for the static image
    print_math_metrics(v_orig, v_enh, label="Static Image")

    # Stack side-by-side (or include visual diff map if requested)
    if img.shape == result.shape:
        if show_diff:
            diff_view = build_diff_view(img, result, gain=diff_gain)
            combined = np.hstack((img, result, diff_view))
        else:
            combined = np.hstack((img, result))
    else:
        combined = result

    # Resize if too large for screen
    max_w = 1920
    if combined.shape[1] > max_w:
        scale = max_w / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    out_dir = get_output_dir(path)
    image_base = os.path.splitext(os.path.basename(str(path)))[0]
    enhanced_path = os.path.join(out_dir, f"{image_base}_enhanced.png")
    comparison_path = os.path.join(out_dir, f"{image_base}_comparison.png")

    cv2.imwrite(enhanced_path, result)
    cv2.imwrite(comparison_path, combined)
    print(f"Saved enhanced image: {enhanced_path}")
    print(f"Saved comparison view: {comparison_path}")

    cv2.imshow("Original vs Enhanced (Press any key to exit)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(source, kappa: float, show_diff: bool, diff_gain: float):
    print(f"Mode: Video | Source: {source}")

    try:
        cap = cv2.VideoCapture(int(source))
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Save outputs under outputs/<source_name>/ for both file and webcam inputs.
    out_dir = get_output_dir(str(source))
    source_name = os.path.splitext(os.path.basename(str(source)))[0] if str(source) != "0" else "webcam"
    writer = None
    output_path = None
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 30.0

    # Initialize Temporal Smoother
    try:
        smoother = he_core.TemporalSmoother()
        print("TemporalSmoother active.")
    except AttributeError:
        smoother = None
        print("Warning: TemporalSmoother not found in C++ module.")

    prev_time = time.time()
    frame_count = 0

    print("\nStarting video loop. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process the frame
        res_frame, v_orig, v_enh = process_frame(frame, smoother=smoother, kappa=kappa)

        # Periodically calculate and print the math (every 30 frames)
        if frame_count % 30 == 0:
            print_math_metrics(v_orig, v_enh, label=f"Video Frame {frame_count}")

        # FPS Calc
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Display side-by-side (optionally with visual diff map)
        if frame.shape == res_frame.shape:
            if show_diff:
                diff_view = build_diff_view(frame, res_frame, gain=diff_gain)
                combined = np.hstack((frame, res_frame, diff_view))
            else:
                combined = np.hstack((frame, res_frame))
        else:
            combined = res_frame

        if writer is None:
            h, w = combined.shape[:2]
            output_path = os.path.join(out_dir, f"{source_name}_enhanced.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, float(source_fps), (w, h))
            if writer.isOpened():
                print(f"Saving processed output to: {output_path} @ {source_fps:.2f} FPS")
            else:
                print("Warning: Could not open VideoWriter. Output file will not be saved.")
                writer = None

        if writer is not None:
            writer.write(combined)

        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Edge-Enhancing Bi-HE (Live)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved normal-speed output video: {output_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Week 3: Final Integrated Pipeline")
    parser.add_argument("input", nargs="?", default="0",
                        help="Path to image/video or '0' for webcam (default: 0)")

    # kappa = sharpening parameter κ. Paper Section 4 sets κ=5 for experiments.
    # Lower values (1-2) give milder enhancement; raise for stronger sharpening.
    # NOTE: '--mu' has been removed. μ is now computed automatically per-frame
    #       as mean(â), matching paper Eq.(21).
    parser.add_argument("--kappa", type=float, default=5.0,
                        help="Sharpening parameter κ (paper uses 5; lower → milder, higher → stronger).")

    parser.add_argument("--show-diff", action="store_true",
                        help="Show a third panel with colorized absolute difference map.")
    parser.add_argument("--diff-gain", type=float, default=5.0,
                        help="Amplification factor for diff-map visibility.")
    parser.add_argument("--threads", type=int, default=4,
                        help="OpenMP thread count for C++ backend (default: 4).")
    args = parser.parse_args()

    # System Lead Constraint: Set C++ OpenMP Thread Count
    try:
        he_core.set_thread_count(args.threads)
        print(f"System: OpenMP Thread Count set to {args.threads}.")
    except AttributeError:
        pass

    input_arg = args.input

    # Route based on file extension
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    _, ext = os.path.splitext(input_arg.lower())

    if input_arg == "0":
        run_video_mode(input_arg, args.kappa, args.show_diff, args.diff_gain)
    elif ext in image_exts:
        run_image_mode(input_arg, args.kappa, args.show_diff, args.diff_gain)
    else:
        run_video_mode(input_arg, args.kappa, args.show_diff, args.diff_gain)


if __name__ == "__main__":
    main()
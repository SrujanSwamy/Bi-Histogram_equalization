"""
Edge-Enhancing Bi-Histogram Equalisation — Week 3 Unified Pipeline
===================================================================
Handles both Static Images & Live Video Feeds.
Includes continuous Mathematical Output (AMBE, Std, CII, DE, PSI).
Implements Temporal Smoothing for video consistency.
Uses Multi-threaded C++ Backend (OpenMP).

Usage:
    python pipeline.py [path/to/image_or_video] (or '0' for webcam)
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
# On Windows, Python 3.8+ no longer searches the system PATH for DLLs by 
# default. We must explicitly add the path to the MinGW runtime libraries
# (which include the OpenMP DLL, libgomp-1.dll) for the C++ module to load.
if sys.platform == 'win32':
    # --- IMPORTANT ---
    # You may need to update this path to match your MSYS2/MinGW installation!
    mingw_bin_path = r"C:\msys64\ucrt64\bin" 
    if os.path.isdir(mingw_bin_path):
        print(f"System: Adding MinGW binary path to DLL search path: {mingw_bin_path}")
        os.add_dll_directory(mingw_bin_path)
    else:
        print(f"Warning: MinGW binary path not found at '{mingw_bin_path}'.")
        print("         The 'he_core' module may fail to load if its dependencies (like libgomp-1.dll) are not found.")

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
def process_frame(frame: np.ndarray, smoother=None):
    """
    Unified processing function for both image and video modes.
    Returns the final BGR image, plus the original and enhanced V channels for math.
    """
    # 1. Convert to HSV and split
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. Compute Statistics (Week 1 Stats)
    hist = he_core.compute_histogram(v)
    I_m = np.mean(v)
    
    # Compute Raw Plateau Limits
    # Note: Assuming your compute_apl_and_clip from week 2 handles this, 
    # we use the standard segment and clip flow.
    # We pass the float I_m to the smoother if available.
    if smoother is not None:
        # Push raw stats, get averaged stats (passing 0 for PLs if calculated inside C++)
        I_m, _, _ = smoother.update_and_get_smoothed(float(I_m), 0.0, 0.0)

    # 3. Generate Modified CDF
    modified_cdf = he_core.compute_apl_and_clip(hist, int(I_m))

    # 4. Integral Images & Busyness Map
    rows, cols = v.shape
    sum_ii, sq_sum_ii = he_core.compute_integral_images(v)
    busyness = he_core.compute_busyness_map(sum_ii, sq_sum_ii, rows, cols)
    
    # 5. Adaptive Guided Filter
    radius = he_core.map_dynamic_radius(busyness, 2, 16) # r_min=2, r_max=16
    a, b_gf = he_core.adaptive_guided_filter(v, sum_ii, sq_sum_ii, radius)
    a_hat = he_core.get_normalized_a(a)

    # 6. Apply Transformation
    v_out = he_core.apply_transformation(v, a_hat, modified_cdf, 0.5, 1.5, float(I_m))

    # 7. Merge & Return
    rgb_out = he_core.merge_hsv_to_rgb(h, s, v_out)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    
    return bgr_out, v, v_out


def run_image_mode(path):
    print(f"Mode: Image | Path: {path}")
    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load image.")
        return

    start_time = time.time()
    result, v_orig, v_enh = process_frame(img, smoother=None)
    elapsed = (time.time() - start_time) * 1000.0
    
    print(f"Processing Complete: {elapsed:.2f} ms")
    
    # Print the exact math for the static image
    print_math_metrics(v_orig, v_enh, label="Static Image")

    # Stack side-by-side
    combined = np.hstack((img, result)) if img.shape == result.shape else result
    
    # Resize if too large for screen
    max_w = 1920
    if combined.shape[1] > max_w:
        scale = max_w / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Original vs Enhanced (Press any key to exit)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(source):
    print(f"Mode: Video | Source: {source}")
    
    try:
        cap = cv2.VideoCapture(int(source))
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

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
        res_frame, v_orig, v_enh = process_frame(frame, smoother=smoother)

        # Periodically calculate and print the math (every 30 frames)
        if frame_count % 30 == 0:
            print_math_metrics(v_orig, v_enh, label=f"Video Frame {frame_count}")

        # FPS Calc
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Display side-by-side
        combined = np.hstack((frame, res_frame)) if frame.shape == res_frame.shape else res_frame

        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Edge-Enhancing Bi-HE (Live)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Week 3: Final Integrated Pipeline")
    parser.add_argument("input", nargs="?", default="0", 
                        help="Path to image/video or '0' for webcam (default: 0)")
    args = parser.parse_args()

    # System Lead Constraint: Set C++ OpenMP Thread Count to 4
    try:
        he_core.set_thread_count(4)
        print("System: OpenMP Thread Count set to 4.")
    except AttributeError:
        pass # Ignore if not compiled in yet

    input_arg = args.input

    # Route based on file extension
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    _, ext = os.path.splitext(input_arg.lower())

    if input_arg == "0":
        run_video_mode(input_arg)
    elif ext in image_exts:
        run_image_mode(input_arg)
    else:
        run_video_mode(input_arg)

if __name__ == "__main__":
    main()
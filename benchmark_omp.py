"""
benchmark_omp.py
================
Engineering benchmark for the hybrid C++/Python implementation of
"Edge-Enhancing Bi-Histogram Equalisation using Guided Image Filter".

This script measures how OpenMP thread count affects end-to-end runtime.
It processes a fixed number of frames from a video and reports:
- Total execution time (ms)
- Average FPS
- Speedup vs 1 thread

Usage:
    python benchmark_omp.py --video test.mp4 --frames 100
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Module loading helper
# -----------------------------------------------------------------------------
def load_he_core() -> object:
    """Load `he_core` from the project build directory with Windows DLL support."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    # Add common CMake output folders to Python module search path.
    for sub in ("build", "build/Release", "build/Debug", "build/RelWithDebInfo", "build/MinSizeRel"):
        candidate = os.path.join(project_root, sub)
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

    # Python 3.8+ on Windows requires explicit DLL directories for dependent runtimes.
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
# Core processing path (single frame)
# -----------------------------------------------------------------------------
def process_frame(he_core: object, frame_bgr: np.ndarray) -> np.ndarray:
    """Run the full custom enhancement pipeline for one BGR frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
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
    return bgr_out


# -----------------------------------------------------------------------------
# Benchmark logic
# -----------------------------------------------------------------------------
def run_benchmark(
    he_core: object, video_path: str, frames_to_process: int, thread_count: int
) -> Tuple[float, float, int]:
    """
    Benchmark one thread configuration.

    Returns:
        total_ms, avg_fps, actual_processed_frames
    """
    he_core.set_thread_count(thread_count)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    processed = 0
    start = time.perf_counter()

    while processed < frames_to_process:
        ok, frame = cap.read()
        if not ok:
            break

        _ = process_frame(he_core, frame)
        processed += 1

    elapsed_s = time.perf_counter() - start
    cap.release()

    if processed == 0:
        raise RuntimeError("No frames were processed. Check the input video.")

    total_ms = elapsed_s * 1000.0
    avg_fps = processed / elapsed_s if elapsed_s > 0 else 0.0
    return total_ms, avg_fps, processed


def print_result_table(rows: List[Dict[str, float]]) -> None:
    """Print a clean, fixed-width benchmark table."""
    print("\nOpenMP Scaling Benchmark (100-frame workload)")
    print("=" * 76)
    print(f"{'Threads':>8} | {'Total Time (ms)':>16} | {'Avg FPS':>10} | {'Speedup vs 1T':>14}")
    print("-" * 76)

    for row in rows:
        print(
            f"{int(row['threads']):>8} | "
            f"{row['total_ms']:>16.2f} | "
            f"{row['avg_fps']:>10.2f} | "
            f"{row['speedup']:>14.2f}x"
        )

    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenMP scaling for he_core.")
    parser.add_argument("--video", default="test.mp4", help="Path to input test video.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process.")
    args = parser.parse_args()

    he_core = load_he_core()

    # Requested benchmark set: 1, 2, 4, and max available threads.
    max_threads = int(he_core.get_omp_max_threads()) if hasattr(he_core, "get_omp_max_threads") else (os.cpu_count() or 4)
    requested = [1, 2, 4, max_threads]

    # Remove duplicates while preserving order.
    thread_configs: List[int] = []
    for t in requested:
        if t > 0 and t not in thread_configs:
            thread_configs.append(t)

    results: List[Dict[str, float]] = []
    baseline_ms = None

    print(f"Input video: {args.video}")
    print(f"Target frames: {args.frames}")

    for threads in thread_configs:
        total_ms, avg_fps, processed = run_benchmark(he_core, args.video, args.frames, threads)

        if baseline_ms is None:
            baseline_ms = total_ms

        speedup = (baseline_ms / total_ms) if total_ms > 0 else 0.0
        results.append(
            {
                "threads": float(threads),
                "total_ms": total_ms,
                "avg_fps": avg_fps,
                "speedup": speedup,
                "processed": float(processed),
            }
        )

    print_result_table(results)


if __name__ == "__main__":
    main()

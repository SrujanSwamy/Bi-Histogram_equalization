import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse


# Required for server-side figure rendering.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = PROJECT_ROOT / "python"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

for sub in ("build", "build/Release", "build/Debug", "build/RelWithDebInfo", "build/MinSizeRel"):
    candidate = PROJECT_ROOT / sub
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

if sys.platform == "win32":
    for dll_dir in (r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin"):
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

from pipeline import build_diff_view, process_frame  # noqa: E402
from generate_visuals import compute_plateau_limits_python  # noqa: E402
from academic_metrics import compute_all_metrics, run_custom_pipeline  # noqa: E402
from benchmark_omp import run_benchmark  # noqa: E402
import he_core  # noqa: E402


KAPPA = 5.0
DIFF_GAIN = 5.0


def encode_bgr_to_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return base64.b64encode(buf).decode("utf-8")


def encode_fig_to_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _to_list_256(hist: Iterable[Any]) -> List[int]:
    arr = np.asarray(hist).reshape(-1)
    if arr.shape[0] != 256:
        arr = np.resize(arr, 256)
    return [int(x) for x in arr]


def _metric_keys_lower(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "ambe": float(metrics["AMBE"]),
        "std": float(metrics["STD"]),
        "cii": float(metrics["CII"]),
        "de": float(metrics["DE"]),
        "psi": float(metrics["PSI"]),
    }


def compute_frame_payload(frame_bgr: np.ndarray, smoother: Optional[Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    enhanced_bgr, v_original, v_enhanced = process_frame(frame_bgr, smoother=smoother, kappa=KAPPA)
    heatmap_bgr = build_diff_view(frame_bgr, enhanced_bgr, gain=DIFF_GAIN)

    metrics_original = {
        "ambe": float(he_core.compute_ambe(v_original)),
        "std": float(he_core.compute_std(v_original)),
    }
    metrics_enhanced = {
        "ambe": float(he_core.compute_ambe(v_enhanced)),
        "std": float(he_core.compute_std(v_enhanced)),
        "cii": float(he_core.compute_cii(v_enhanced)),
        "de": float(he_core.compute_de(v_enhanced)),
        "psi": float(he_core.compute_psi(v_enhanced)),
    }

    hist_original = _to_list_256(he_core.compute_histogram(v_original))
    hist_enhanced = _to_list_256(he_core.compute_histogram(v_enhanced))

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "original_img": encode_bgr_to_b64(frame_bgr),
        "enhanced_img": encode_bgr_to_b64(enhanced_bgr),
        "heatmap_img": encode_bgr_to_b64(heatmap_bgr),
        "metrics": {
            "original": metrics_original,
            "enhanced": metrics_enhanced,
        },
        "histograms": {
            "original": hist_original,
            "enhanced": hist_enhanced,
        },
        "processing_time_ms": float(elapsed_ms),
    }


def run_generate_visuals(bgr_img: np.ndarray) -> Dict[str, str]:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hist_cpp = he_core.compute_histogram(v)
    hist = np.asarray(hist_cpp).reshape(-1)

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
    v_out = he_core.apply_transformation(v, a_hat, modified_cdf, mu, 5.0, i_m)

    fig1 = plt.figure(figsize=(12, 7))
    ax1 = fig1.add_subplot(111)
    x = np.arange(256)
    ax1.plot(x, hist, color="royalblue", linewidth=1.6, label="Original Histogram")
    ax1.axvline(i_m, color="darkred", linestyle="--", linewidth=2.0, label=f"Mean Intensity I_m = {i_m:.0f}")
    ax1.hlines(pl_l, 0, i_m, colors="seagreen", linestyles="-.", linewidth=2.0, label=f"PL_L = {pl_l:.2f}")
    ax1.hlines(pl_h, i_m, 255, colors="orange", linestyles="-.", linewidth=2.0, label=f"PL_H = {pl_h:.2f}")
    ax1.set_title("Bi-Histogram Segmentation and Plateau Clipping", fontsize=14, pad=12)
    ax1.set_xlabel("Intensity Level", fontsize=12)
    ax1.set_ylabel("Pixel Count", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")
    fig1.tight_layout()

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

    hist_in = cv2.calcHist([v], [0], None, [256], [0, 256]).flatten()
    hist_out = cv2.calcHist([v_out], [0], None, [256], [0, 256]).flatten()
    fig3 = plt.figure(figsize=(12, 7))
    ax3 = fig3.add_subplot(111)
    ax3.bar(np.arange(256), hist_in, width=1.0, color="steelblue", alpha=0.55, label="Input Intensity Frequency")
    ax3.bar(np.arange(256), hist_out, width=1.0, color="crimson", alpha=0.45, label="Enhanced Intensity Frequency")
    ax3.set_title("Intensity vs Frequency (Bar Plot): Input vs Enhanced", fontsize=14, pad=12)
    ax3.set_xlabel("Intensity", fontsize=12)
    ax3.set_ylabel("Frequency (Pixel Count)", fontsize=12)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper right")
    fig3.tight_layout()

    return {
        "bi_histogram": encode_fig_to_b64(fig1),
        "busyness_map": encode_fig_to_b64(fig2),
        "intensity_frequency": encode_fig_to_b64(fig3),
    }


def run_academic_metrics(bgr_img: np.ndarray) -> Dict[str, Dict[str, float]]:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _h, _s, v_original = cv2.split(hsv)
    v_he = cv2.equalizeHist(v_original)
    custom_out = run_custom_pipeline(he_core, bgr_img)
    return {
        "original": _metric_keys_lower(compute_all_metrics(he_core, v_original)),
        "opencv_he": _metric_keys_lower(compute_all_metrics(he_core, v_he)),
        "custom": _metric_keys_lower(compute_all_metrics(he_core, custom_out["v_custom"])),
    }


def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload.file.seek(0)
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name


def create_video_writer(output_dir: Path, base_name: str, fps: float, size: Tuple[int, int]):
    # Prefer WebM codecs first for browser playback; fallback to MP4 if needed.
    candidates = [
        ("webm", "VP90"),
        ("webm", "VP80"),
        ("mp4", "mp4v"),
    ]

    for ext, codec in candidates:
        file_path = output_dir / f"{base_name}.{ext}"
        writer = cv2.VideoWriter(str(file_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, file_path
        writer.release()

    raise RuntimeError("Failed to initialize any supported output video codec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.smoother = he_core.TemporalSmoother()
    app.state.webcam = None
    app.state.webcam_active = False
    yield
    webcam = app.state.webcam
    if webcam is not None and webcam.isOpened():
        webcam.release()


app = FastAPI(title="EEBHE Dashboard API", lifespan=lifespan)
app.mount("/frontend", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR), html=False), name="outputs")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "omp_threads": int(he_core.get_omp_max_threads()),
    }


@app.post("/process/image")
def process_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_path = save_upload_to_temp(file)
    try:
        bgr = cv2.imread(temp_path)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        frame_payload = compute_frame_payload(bgr, smoother=None)
        frame_payload["visuals"] = run_generate_visuals(bgr)
        frame_payload["academic_comparison"] = run_academic_metrics(bgr)
        return frame_payload
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/process/video")
async def process_video(file: UploadFile = File(...)) -> EventSourceResponse:
    temp_path = save_upload_to_temp(file)
    video_name = Path(file.filename or "video").stem or "video"

    async def event_generator():
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            yield {"data": json.dumps({"error": "Invalid video file"})}
            return

        smoother = he_core.TemporalSmoother()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        frame_idx = 0
        first_frame: Optional[np.ndarray] = None
        first_frame_png_path: Optional[str] = None
        output_video_dir = OUTPUTS_DIR / video_name
        output_video_dir.mkdir(parents=True, exist_ok=True)

        writer_original = None
        writer_enhanced = None
        writer_heatmap = None

        original_video_path: Optional[Path] = None
        enhanced_video_path: Optional[Path] = None
        heatmap_video_path: Optional[Path] = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if first_frame is None:
                    first_frame = frame.copy()
                    h, w = frame.shape[:2]
                    writer_original, original_video_path = create_video_writer(output_video_dir, "original", fps, (w, h))
                    writer_enhanced, enhanced_video_path = create_video_writer(output_video_dir, "enhanced", fps, (w, h))
                    writer_heatmap, heatmap_video_path = create_video_writer(output_video_dir, "heatmap", fps, (w, h))

                start = time.perf_counter()
                enhanced_bgr, v_original, v_enhanced = process_frame(frame, smoother=smoother, kappa=KAPPA)
                heatmap_bgr = build_diff_view(frame, enhanced_bgr, gain=DIFF_GAIN)

                if writer_original is not None:
                    writer_original.write(frame)
                if writer_enhanced is not None:
                    writer_enhanced.write(enhanced_bgr)
                if writer_heatmap is not None:
                    writer_heatmap.write(heatmap_bgr)

                metrics_original = {
                    "ambe": float(he_core.compute_ambe(v_original)),
                    "std": float(he_core.compute_std(v_original)),
                }
                metrics_enhanced = {
                    "ambe": float(he_core.compute_ambe(v_enhanced)),
                    "std": float(he_core.compute_std(v_enhanced)),
                    "cii": float(he_core.compute_cii(v_enhanced)),
                    "de": float(he_core.compute_de(v_enhanced)),
                    "psi": float(he_core.compute_psi(v_enhanced)),
                }

                frame_idx += 1
                payload = {
                    "frame_number": frame_idx,
                    "total_frames": total_frames,
                    "fps": fps,
                    "original_img": encode_bgr_to_b64(frame),
                    "enhanced_img": encode_bgr_to_b64(enhanced_bgr),
                    "heatmap_img": encode_bgr_to_b64(heatmap_bgr),
                    "metrics": {
                        "original": metrics_original,
                        "enhanced": metrics_enhanced,
                    },
                    "histograms": {
                        "original": _to_list_256(he_core.compute_histogram(v_original)),
                        "enhanced": _to_list_256(he_core.compute_histogram(v_enhanced)),
                    },
                    "processing_time_ms": float((time.perf_counter() - start) * 1000.0),
                }
                yield {"data": json.dumps(payload)}
                await asyncio.sleep(0)

            if first_frame is not None:
                if writer_original is not None:
                    writer_original.release()
                    writer_original = None
                if writer_enhanced is not None:
                    writer_enhanced.release()
                    writer_enhanced = None
                if writer_heatmap is not None:
                    writer_heatmap.release()
                    writer_heatmap = None

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as first_png:
                    first_frame_png_path = first_png.name
                cv2.imwrite(first_frame_png_path, first_frame)
                first_frame_for_reports = cv2.imread(first_frame_png_path)
                if first_frame_for_reports is None:
                    first_frame_for_reports = first_frame

                done_payload = {
                    "done": True,
                    "fps": fps,
                    "visuals": run_generate_visuals(first_frame_for_reports),
                    "academic_comparison": run_academic_metrics(first_frame_for_reports),
                    "video_urls": {
                        "original": f"/outputs/{video_name}/{original_video_path.name}" if original_video_path else None,
                        "enhanced": f"/outputs/{video_name}/{enhanced_video_path.name}" if enhanced_video_path else None,
                        "heatmap": f"/outputs/{video_name}/{heatmap_video_path.name}" if heatmap_video_path else None,
                    },
                }
                yield {"data": json.dumps(done_payload)}
            else:
                yield {"data": json.dumps({"done": True, "visuals": {}, "academic_comparison": {}})}
        finally:
            if writer_original is not None:
                writer_original.release()
            if writer_enhanced is not None:
                writer_enhanced.release()
            if writer_heatmap is not None:
                writer_heatmap.release()
            cap.release()
            if first_frame_png_path and os.path.exists(first_frame_png_path):
                os.remove(first_frame_png_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return EventSourceResponse(event_generator())


@app.get("/webcam/stream")
async def webcam_stream(request: Request) -> EventSourceResponse:
    if app.state.webcam is None or not app.state.webcam.isOpened():
        app.state.webcam = cv2.VideoCapture(0)

    if not app.state.webcam.isOpened():
        raise HTTPException(status_code=500, detail="Unable to open webcam")

    app.state.webcam_active = True

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected() or not app.state.webcam_active:
                    break
                cam = app.state.webcam
                if cam is None or not cam.isOpened():
                    break
                ok, frame = cam.read()
                if not ok:
                    await asyncio.sleep(0.05)
                    continue
                payload = compute_frame_payload(frame, smoother=app.state.smoother)
                yield {"data": json.dumps(payload)}
                await asyncio.sleep(0)
        finally:
            _release_webcam()

    return EventSourceResponse(event_generator())


def _release_webcam():
    cam = app.state.webcam
    if cam is not None:
        try:
            if cam.isOpened():
                cam.release()
        except Exception:
            pass
    app.state.webcam = None
    app.state.webcam_active = False


@app.post("/webcam/stop")
def webcam_stop() -> Dict[str, str]:
    app.state.webcam_active = False
    _release_webcam()
    return {"status": "stopped"}


@app.post("/benchmark")
def benchmark(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_path = save_upload_to_temp(file)
    try:
        max_threads = int(he_core.get_omp_max_threads()) if hasattr(he_core, "get_omp_max_threads") else (os.cpu_count() or 4)
        requested = [1, 2, 4, max_threads]
        thread_configs: List[int] = []
        for t in requested:
            if t > 0 and t not in thread_configs:
                thread_configs.append(t)

        rows: List[Dict[str, Any]] = []
        baseline_ms = None

        for threads in thread_configs:
            total_ms, avg_fps, processed = run_benchmark(
                he_core,
                temp_path,
                frames_to_process=100,
                thread_count=threads,
            )
            if baseline_ms is None:
                baseline_ms = total_ms
            speedup = (baseline_ms / total_ms) if total_ms > 0 else 0.0
            rows.append(
                {
                    "threads": int(threads),
                    "total_ms": float(total_ms),
                    "avg_fps": float(avg_fps),
                    "speedup": float(speedup),
                    "processed": int(processed),
                }
            )

        return {
            "results": rows,
            "max_threads": int(max_threads),
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
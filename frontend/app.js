const state = {
  mode: "image",
  streaming: false,
  eventSource: null,
  videoFile: null,
  histChart: null,
  benchChart: null,
};

const el = {
  tabs: Array.from(document.querySelectorAll(".tab")),
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  webcamActions: document.getElementById("webcamActions"),
  webcamToggle: document.getElementById("webcamToggle"),
  statusPill: document.getElementById("statusPill"),
  statusLabel: document.getElementById("statusLabel"),
  progressWrap: document.getElementById("progressWrap"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  origImg: document.getElementById("origImg"),
  enhImg: document.getElementById("enhImg"),
  heatImg: document.getElementById("heatImg"),
  origVideo: document.getElementById("origVideo"),
  enhVideo: document.getElementById("enhVideo"),
  heatVideo: document.getElementById("heatVideo"),
  origRes: document.getElementById("origRes"),
  enhRes: document.getElementById("enhRes"),
  heatRes: document.getElementById("heatRes"),
  origCard: document.getElementById("origCard"),
  enhCard: document.getElementById("enhCard"),
  heatCard: document.getElementById("heatCard"),
  visualsPanel: document.getElementById("visualsPanel"),
  visualGrid: document.getElementById("visualGrid"),
  academicPanel: document.getElementById("academicPanel"),
  benchmarkPanel: document.getElementById("benchmarkPanel"),
  visBi: document.getElementById("visBi"),
  visBusy: document.getElementById("visBusy"),
  visFreq: document.getElementById("visFreq"),
  metricsBody: document.getElementById("metricsBody"),
  academicBody: document.getElementById("academicBody"),
  benchBody: document.getElementById("benchBody"),
  lightbox: document.getElementById("lightbox"),
  lightboxImg: document.getElementById("lightboxImg"),
  closeLightbox: document.getElementById("closeLightbox"),
  toasts: document.getElementById("toasts"),
};

const metricRows = [
  { key: "ambe", label: "Mean Brightness (AMBE)" },
  { key: "std", label: "Std Deviation" },
  { key: "cii", label: "Contrast (CII)" },
  { key: "de", label: "Entropy (DE)" },
  { key: "psi", label: "Sharpness (PSI)" },
];

function init() {
  initCharts();
  renderMetricsSkeleton();
  bindEvents();
  switchMode("image");
  setStatus("IDLE");
}

function bindEvents() {
  el.tabs.forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  el.dropZone.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    handleSelectedFile(file);
    el.fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) => {
    el.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      el.dropZone.classList.add("drag");
    });
  });

  ["dragleave", "drop"].forEach((ev) => {
    el.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      if (ev === "drop") {
        const file = e.dataTransfer?.files?.[0];
        if (file) handleSelectedFile(file);
      }
      el.dropZone.classList.remove("drag");
    });
  });

  el.webcamToggle.addEventListener("click", () => {
    if (state.streaming) stopWebcam();
    else startWebcam();
  });

  [el.visBi, el.visBusy, el.visFreq].forEach((img) => {
    img.addEventListener("click", () => openLightbox(img.src));
  });

  el.closeLightbox.addEventListener("click", closeLightbox);
  el.lightbox.addEventListener("click", (e) => {
    if (e.target === el.lightbox) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
}

function setStatus(label) {
  el.statusLabel.textContent = label;
  el.statusPill.classList.remove("pulse");
  const dot = el.statusPill.querySelector(".dot");
  const map = {
    IDLE: "#555",
    PROCESSING: "var(--amber)",
    STREAMING: "var(--green)",
    COMPLETE: "var(--green)",
    ERROR: "var(--red)",
  };
  dot.style.background = map[label] || "#555";
  if (label === "PROCESSING" || label === "STREAMING") {
    el.statusPill.classList.add("pulse");
  }
}

function switchMode(mode) {
  stopStreams();
  state.mode = mode;
  state.videoFile = null;
  resetViewerMedia();

  el.tabs.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  el.webcamActions.style.display = mode === "webcam" ? "flex" : "none";
  el.dropZone.style.display = mode === "webcam" ? "none" : "grid";

  if (mode === "image") {
    el.fileInput.accept = ".jpg,.jpeg,.png,.bmp,.tiff,.tif";
    el.dropZone.firstElementChild.children[0].textContent = "Drag and drop image here";
  }

  if (mode === "video") {
    el.fileInput.accept = ".mp4,.avi,.mov";
    el.dropZone.firstElementChild.children[0].textContent = "Drag and drop video here";
  }

  if (mode === "webcam") {
    hidePanelsForLive();
  }

  el.progressWrap.style.display = "none";
  if (mode !== "video") {
    el.benchmarkPanel.classList.add("hidden");
  }

  setStatus("IDLE");
}

function stopStreams() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.streaming = false;
  el.webcamToggle.textContent = "▶ START STREAM";
}

function showLoadingViewer(show) {
  [el.origCard, el.enhCard, el.heatCard].forEach((card) => {
    card.classList.toggle("loading", show);
  });
}

async function handleSelectedFile(file) {
  hideTransientSections();
  if (state.mode === "image") {
    await handleImageUpload(file);
    return;
  }
  if (state.mode === "video") {
    await handleVideoUpload(file);
  }
}

async function handleImageUpload(file) {
  try {
    setStatus("PROCESSING");
    showLoadingViewer(true);
    const form = new FormData();
    form.append("file", file);

    const res = await fetch("/process/image", { method: "POST", body: form });
    if (!res.ok) {
      throw new Error("Image processing failed");
    }

    const payload = await res.json();
    updateViewerPanels(payload.original_img, payload.enhanced_img, payload.heatmap_img);
    updateMetricsTable(payload.metrics, payload.processing_time_ms, false);
    updateHistChart(payload.histograms);
    updateVisualsPanel(payload.visuals);
    updateAcademicPanel(payload.academic_comparison);
    el.benchmarkPanel.classList.add("hidden");
    setStatus("COMPLETE");
    showToast("Image processing complete", "success");
  } catch (err) {
    setStatus("ERROR");
    showToast(err.message || "Image processing error", "error");
  } finally {
    showLoadingViewer(false);
  }
}

async function handleVideoUpload(file) {
  state.videoFile = file;
  try {
    setStatus("PROCESSING");
    showLoadingViewer(true);
    resetViewerMedia();
    el.progressWrap.style.display = "block";
    el.progressFill.style.width = "0%";

    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/process/video", { method: "POST", body: form });
    if (!response.ok || !response.body) {
      throw new Error("Video processing stream failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let firstFrame = true;
    let donePayload = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        const payload = JSON.parse(raw);
        if (payload.error) {
          throw new Error(payload.error);
        }

        if (payload.done) {
          donePayload = payload;
        } else {
          // Remove loading overlay on first frame so subsequent frames are visible
          if (firstFrame) {
            showLoadingViewer(false);
            firstFrame = false;
          }
          updateViewerPanels(payload.original_img, payload.enhanced_img, payload.heatmap_img);
          updateMetricsTable(payload.metrics, payload.processing_time_ms, false);
          updateHistChart(payload.histograms);
          updateProgress(payload.frame_number, payload.total_frames || payload.frame_number);
        }
      }
    }

    if (donePayload) {
      if (donePayload.video_urls) {
        showPlaybackVideos(donePayload.video_urls, donePayload.fps);
      }
      updateVisualsPanel(donePayload.visuals);
      updateAcademicPanel(donePayload.academic_comparison);
      setStatus("COMPLETE");
      showLoadingViewer(false);
      await runBenchmark(state.videoFile);
    }
  } catch (err) {
    setStatus("ERROR");
    showToast(err.message || "Video processing error", "error");
  } finally {
    showLoadingViewer(false);
  }
}

function startWebcam() {
  if (state.streaming) return;
  hidePanelsForLive();
  setStatus("STREAMING");
  state.streaming = true;
  el.webcamToggle.textContent = "■ STOP STREAM";

  state.eventSource = new EventSource("/webcam/stream");
  state.eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      updateViewerPanels(payload.original_img, payload.enhanced_img, payload.heatmap_img);
      updateMetricsTable(payload.metrics, payload.processing_time_ms, true);
      updateHistChart(payload.histograms);
    } catch {
      setStatus("ERROR");
    }
  };

  state.eventSource.onerror = () => {
    showToast("Webcam stream disconnected", "error");
    stopWebcam();
    setStatus("ERROR");
  };
}

function stopWebcam() {
  stopStreams();
  setStatus("IDLE");
  // Tell server to release the camera device — close EventSource alone is not enough
  // because the server SSE generator loops independently and holds cv2.VideoCapture open.
  fetch("/webcam/stop", { method: "POST" }).catch(() => {});
}

async function runBenchmark(file) {
  if (!file) return;
  try {
    setStatus("PROCESSING");
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/benchmark", { method: "POST", body: form });
    if (!res.ok) {
      throw new Error("Benchmark request failed");
    }
    const payload = await res.json();
    updateBenchmarkPanel(payload.results || []);
    setStatus("COMPLETE");
  } catch (err) {
    showToast(err.message || "Benchmark failed", "error");
    setStatus("ERROR");
  }
}

function updateViewerPanels(orig, enh, heat) {
  setImage(el.origImg, orig, el.origRes);
  setImage(el.enhImg, enh, el.enhRes);
  setImage(el.heatImg, heat, el.heatRes);
}

function resetViewerMedia() {
  [
    [el.origImg, el.origVideo],
    [el.enhImg, el.enhVideo],
    [el.heatImg, el.heatVideo],
  ].forEach(([img, video]) => {
    img.style.display = "none";
    img.removeAttribute("src");

    video.pause();
    video.style.display = "none";
    video.removeAttribute("src");
    video.load();

    img.parentElement.querySelector(".viewer-empty").style.display = "grid";
  });
}

function setImage(node, b64, resNode) {
  const videoNode = node === el.origImg ? el.origVideo : node === el.enhImg ? el.enhVideo : el.heatVideo;
  videoNode.pause();
  videoNode.style.display = "none";
  videoNode.removeAttribute("src");
  videoNode.load();

  const src = `data:image/jpeg;base64,${b64}`;
  node.onload = () => {
    resNode.textContent = `${node.naturalWidth}x${node.naturalHeight} px`;
  };
  node.src = src;
  node.style.display = "block";
  node.parentElement.querySelector(".viewer-empty").style.display = "none";
}

function showPlaybackVideos(videoUrls, fps) {
  const stamp = Date.now().toString();
  setVideo(el.origVideo, el.origImg, el.origRes, videoUrls.original, fps, stamp);
  setVideo(el.enhVideo, el.enhImg, el.enhRes, videoUrls.enhanced, fps, stamp);
  setVideo(el.heatVideo, el.heatImg, el.heatRes, videoUrls.heatmap, fps, stamp);
}

function setVideo(videoNode, imgNode, resNode, url, fps, stamp) {
  if (!url) return;

  imgNode.style.display = "none";
  imgNode.removeAttribute("src");

  const cacheBusted = `${url}${url.includes("?") ? "&" : "?"}t=${stamp}`;
  videoNode.src = cacheBusted;
  videoNode.style.display = "block";
  videoNode.parentElement.querySelector(".viewer-empty").style.display = "none";

  videoNode.onloadedmetadata = () => {
    const width = videoNode.videoWidth;
    const height = videoNode.videoHeight;
    const fpsLabel = fps ? ` @ ${Number(fps).toFixed(2)} fps` : "";
    resNode.textContent = `${width}x${height} px${fpsLabel}`;
  };

  videoNode.play().catch(() => {});
}

function renderMetricsSkeleton() {
  el.metricsBody.innerHTML = "";
  metricRows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.label}</td>
      <td class="value" id="orig-${row.key}">-</td>
      <td class="value" id="enh-${row.key}">-</td>
    `;
    el.metricsBody.appendChild(tr);
  });

  const tr = document.createElement("tr");
  tr.innerHTML = `<td>Processing Time</td><td class="value" colspan="2" id="proc-time" style="color: var(--green);">-</td>`;
  el.metricsBody.appendChild(tr);
}

function updateMetricsTable(metrics, processingTime, isWebcam) {
  if (!metrics || !metrics.original || !metrics.enhanced) return;

  metricRows.forEach((row) => {
    const o = Number(metrics.original[row.key] || 0);
    const e = Number(metrics.enhanced[row.key] || 0);
    const oCell = document.getElementById(`orig-${row.key}`);
    const eCell = document.getElementById(`enh-${row.key}`);

    oCell.textContent = o.toFixed(4);
    eCell.innerHTML = `${e.toFixed(4)} ${deltaIndicator(row.key, o, e)}`;

    if (!isWebcam) {
      oCell.classList.remove("flash");
      eCell.classList.remove("flash");
      void oCell.offsetWidth;
      oCell.classList.add("flash");
      eCell.classList.add("flash");
    }
  });

  document.getElementById("proc-time").textContent = `${Number(processingTime || 0).toFixed(2)} ms`;
}

function deltaIndicator(key, original, enhanced) {
  const improve = key === "ambe" ? enhanced < original : enhanced > original;
  return `<span class="${improve ? "delta-up" : "delta-down"}">${improve ? "▲" : "▼"}</span>`;
}

function initCharts() {
  const labels = Array.from({ length: 256 }, (_, i) => i.toString());
  state.histChart = new Chart(document.getElementById("histChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Original",
          data: new Array(256).fill(0),
          backgroundColor: "rgba(77, 158, 255, 0.5)",
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
        {
          label: "Enhanced",
          data: new Array(256).fill(0),
          backgroundColor: "rgba(0, 229, 160, 0.5)",
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
      ],
    },
    options: {
      animation: false,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: "Intensity Distribution", color: "#e8e8f0" },
        legend: { labels: { color: "#e8e8f0" } },
      },
      scales: {
        x: {
          ticks: {
            color: "#7a7a9a",
            callback: (v, i) => (i % 32 === 0 ? labels[i] : ""),
            font: { family: "IBM Plex Mono", size: 11 },
          },
          title: { display: true, text: "Intensity Level", color: "#e8e8f0", font: { family: "IBM Plex Mono", size: 11 } },
          grid: { color: "#2a2a3a" },
        },
        y: {
          ticks: { color: "#7a7a9a", font: { family: "IBM Plex Mono", size: 11 } },
          title: { display: true, text: "Pixel Count", color: "#e8e8f0", font: { family: "IBM Plex Mono", size: 11 } },
          grid: { color: "#2a2a3a" },
        },
      },
    },
  });

  const speedLabelPlugin = {
    id: "speedLabelPlugin",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.fillStyle = "#e8e8f0";
      ctx.font = "12px IBM Plex Mono";
      meta.data.forEach((bar, idx) => {
        const val = chart.data.datasets[0].data[idx] || 0;
        ctx.fillText(`${Number(val).toFixed(2)}x`, bar.x + 6, bar.y + 4);
      });
      ctx.restore();
    },
  };

  state.benchChart = new Chart(document.getElementById("benchChart"), {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Speedup", data: [], backgroundColor: "rgba(0, 229, 160, 0.8)" }] },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Thread Scaling Speedup", color: "#e8e8f0" },
      },
      scales: {
        x: { min: 0, grid: { color: "#2a2a3a" }, ticks: { color: "#7a7a9a" }, title: { display: true, text: "Speedup (x)", color: "#e8e8f0" } },
        y: { grid: { color: "#2a2a3a" }, ticks: { color: "#7a7a9a" } },
      },
    },
    plugins: [speedLabelPlugin],
  });
}

function updateHistChart(histograms) {
  if (!histograms || !state.histChart) return;
  state.histChart.data.datasets[0].data = histograms.original || [];
  state.histChart.data.datasets[1].data = histograms.enhanced || [];
  state.histChart.update("none");
}

function updateProgress(frame, total) {
  const pct = total > 0 ? (frame / total) * 100 : 0;
  el.progressFill.style.width = `${pct}%`;
  el.progressText.textContent = `PROCESSING FRAME ${frame} / ${total}`;
}

function updateVisualsPanel(visuals) {
  if (!visuals) return;
  el.visBi.src = `data:image/png;base64,${visuals.bi_histogram || ""}`;
  el.visBusy.src = `data:image/png;base64,${visuals.busyness_map || ""}`;
  el.visFreq.src = `data:image/png;base64,${visuals.intensity_frequency || ""}`;
  el.visualsPanel.classList.remove("hidden");
  el.visualGrid.classList.add("show");
}

function updateAcademicPanel(cmp) {
  if (!cmp) return;
  const rows = [
    ["AMBE", "ambe", true],
    ["STD", "std", false],
    ["CII", "cii", false],
    ["DE", "de", false],
    ["PSI", "psi", false],
  ];

  el.academicBody.innerHTML = "";
  rows.forEach(([label, key, lowerIsBest]) => {
    const values = [cmp.original?.[key] ?? 0, cmp.opencv_he?.[key] ?? 0, cmp.custom?.[key] ?? 0].map(Number);
    const best = lowerIsBest ? Math.min(...values) : Math.max(...values);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${label}</td>
      <td class="value ${values[0] === best ? "best" : ""}">${values[0].toFixed(4)}</td>
      <td class="value ${values[1] === best ? "best" : ""}">${values[1].toFixed(4)}</td>
      <td class="value ${values[2] === best ? "best" : ""}">${values[2].toFixed(4)}</td>
    `;
    el.academicBody.appendChild(tr);
  });
  el.academicPanel.classList.remove("hidden");
}

function updateBenchmarkPanel(results) {
  if (!results?.length) return;

  const bestSpeed = Math.max(...results.map((r) => Number(r.speedup || 0)));
  el.benchBody.innerHTML = "";
  results.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = Number(r.speedup) === bestSpeed ? "best" : "";
    tr.innerHTML = `
      <td class="value">${r.threads}</td>
      <td class="value">${Number(r.total_ms).toFixed(2)}</td>
      <td class="value">${Number(r.avg_fps).toFixed(2)}</td>
      <td class="value">${Number(r.speedup).toFixed(2)}x</td>
    `;
    el.benchBody.appendChild(tr);
  });

  state.benchChart.data.labels = results.map((r) => `${r.threads} thread${r.threads > 1 ? "s" : ""}`);
  state.benchChart.data.datasets[0].data = results.map((r) => Number(r.speedup || 0));
  state.benchChart.update("none");
  el.benchmarkPanel.classList.remove("hidden");
}

function hideTransientSections() {
  el.visualsPanel.classList.add("hidden");
  el.academicPanel.classList.add("hidden");
  if (state.mode !== "video") {
    el.benchmarkPanel.classList.add("hidden");
  }
}

function hidePanelsForLive() {
  el.visualsPanel.classList.add("hidden");
  el.academicPanel.classList.add("hidden");
  el.benchmarkPanel.classList.add("hidden");
}

function showToast(msg, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.textContent = msg;
  el.toasts.appendChild(item);
  setTimeout(() => item.remove(), 4000);
}

function openLightbox(src) {
  if (!src) return;
  el.lightboxImg.src = src;
  el.lightbox.classList.add("open");
}

function closeLightbox() {
  el.lightbox.classList.remove("open");
  el.lightboxImg.src = "";
}

init();
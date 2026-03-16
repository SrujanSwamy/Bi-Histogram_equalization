// ============================================================================
//  Member 1 — Data & Color-Space + Evaluation Metrics
// ============================================================================
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common.h"

namespace py = pybind11;

// ============================================================================
//  V-Channel Extraction
// ============================================================================

// Extract the HSV Value channel: V = max(R, G, B).
// Accepts 2-D grayscale (H×W) or 3-D colour (H×W×3, any channel order).
inline npy<uint8_t> extract_v_channel(npy<uint8_t> img) {
    py::buffer_info buf = img.request();

    if (buf.ndim == 2) {
        // Grayscale — return a copy
        int rows = static_cast<int>(buf.shape[0]);
        int cols = static_cast<int>(buf.shape[1]);
        npy<uint8_t> result({buf.shape[0], buf.shape[1]});
        auto rbuf = result.request();
        std::memcpy(rbuf.ptr, buf.ptr, static_cast<size_t>(rows) * cols);
        return result;
    }

    if (buf.ndim == 3 && buf.shape[2] == 3) {
        int rows = static_cast<int>(buf.shape[0]);
        int cols = static_cast<int>(buf.shape[1]);
        const uint8_t* src = static_cast<const uint8_t*>(buf.ptr);

        npy<uint8_t> result({buf.shape[0], buf.shape[1]});
        auto rbuf = result.request();
        uint8_t* dst = static_cast<uint8_t*>(rbuf.ptr);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int idx3 = (i * cols + j) * 3;
                dst[i * cols + j] =
                    std::max({src[idx3], src[idx3 + 1], src[idx3 + 2]});
            }
        }
        return result;
    }

    throw std::runtime_error(
        "extract_v_channel: input must be 2-D grayscale or 3-D with 3 channels.");
}

// ============================================================================
//  Evaluation Metrics
// ============================================================================

// Mean brightness (baseline for AMBE = |mean_orig − mean_enhanced|).
inline double compute_ambe(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_ambe: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int N = static_cast<int>(buf.shape[0] * buf.shape[1]);

    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += ptr[i];
    return sum / N;
}

// Global contrast — standard deviation.
inline double compute_std(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_std: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int N = static_cast<int>(buf.shape[0] * buf.shape[1]);

    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += ptr[i];
    mean /= N;

    double var = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = ptr[i] - mean;
        var += d * d;
    }
    return std::sqrt(var / N);
}

// Contrast Improvement Index — average local contrast in a 3×3 window.
//   C_loc = (max − min) / (max + min)      (skipped when max + min == 0)
inline double compute_cii(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_cii: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    double cii_sum = 0.0;
    int    count   = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint8_t lo = 255, hi = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ni = clamp_idx(i + di, 0, rows - 1);
                    int nj = clamp_idx(j + dj, 0, cols - 1);
                    uint8_t val = ptr[ni * cols + nj];
                    lo = std::min(lo, val);
                    hi = std::max(hi, val);
                }
            }
            double denom = static_cast<double>(hi) + lo;
            if (denom > 0.0)
                cii_sum += (static_cast<double>(hi) - lo) / denom;
            ++count;
        }
    }
    return (count > 0) ? cii_sum / count : 0.0;
}

// Discrete Entropy:  H = −Σ p_i · log₂(p_i)
inline double compute_de(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_de: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int N = static_cast<int>(buf.shape[0] * buf.shape[1]);

    int hist[256] = {};
    for (int i = 0; i < N; ++i) hist[ptr[i]]++;

    double entropy = 0.0;
    for (int k = 0; k < 256; ++k) {
        if (hist[k] > 0) {
            double p = static_cast<double>(hist[k]) / N;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// Perceptual Sharpness Index (simplified) — variance of local gradient
// magnitudes computed via central differences.
inline double compute_psi(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_psi: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    if (rows < 3 || cols < 3) return 0.0;

    // Interior pixels only (1 .. rows-2, 1 .. cols-2)
    int M = (rows - 2) * (cols - 2);
    std::vector<double> grads(M);
    int idx = 0;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            double gx = static_cast<double>(ptr[i * cols + (j + 1)]) -
                        static_cast<double>(ptr[i * cols + (j - 1)]);
            double gy = static_cast<double>(ptr[(i + 1) * cols + j]) -
                        static_cast<double>(ptr[(i - 1) * cols + j]);
            grads[idx++] = std::sqrt(gx * gx + gy * gy);
        }
    }

    double mean = 0.0;
    for (double g : grads) mean += g;
    mean /= M;

    double var = 0.0;
    for (double g : grads) {
        double d = g - mean;
        var += d * d;
    }
    return var / M;
}

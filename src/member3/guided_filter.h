// ============================================================================
//  Member 3 — Guided Image Filter Coefficients (OpenMP-parallelised)
// ============================================================================
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

#include <omp.h>

#include "common.h"

namespace py = pybind11;

// ============================================================================
//  Guided Filter Coefficients
// ============================================================================
//
// Self-guided filter (guidance = input = V channel), 3×3 window (radius = 1).
// For each pixel k with window ω_k:
//   μ_k   = mean(I ∈ ω_k)
//   σ²_k  = var(I ∈ ω_k)        (local variance)
//   a_k   = σ²_k / (σ²_k + ε)
//   b_k   = μ_k · (1 − a_k)
//
// Returns (a_coeff, b_coeff) as 2-D double arrays.
inline py::tuple compute_gf_coefficients(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error(
            "compute_gf_coefficients: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    // Regularisation parameter ε  (≈ 0.01 × 255²)
    const double eps = 0.01 * 255.0 * 255.0;
    const int r = 1;  // radius → 3×3 window

    npy<double> a_out({buf.shape[0], buf.shape[1]});
    npy<double> b_out({buf.shape[0], buf.shape[1]});
    double* a_ptr = static_cast<double*>(a_out.request().ptr);
    double* b_ptr = static_cast<double*>(b_out.request().ptr);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sum    = 0.0;
            double sum_sq = 0.0;
            int    cnt    = 0;

            for (int di = -r; di <= r; ++di) {
                for (int dj = -r; dj <= r; ++dj) {
                    int ni = clamp_idx(i + di, 0, rows - 1);
                    int nj = clamp_idx(j + dj, 0, cols - 1);
                    double val = static_cast<double>(ptr[ni * cols + nj]);
                    sum    += val;
                    sum_sq += val * val;
                    ++cnt;
                }
            }

            double mu     = sum / cnt;
            double sigma2 = sum_sq / cnt - mu * mu;

            double a = sigma2 / (sigma2 + eps);
            double b = mu * (1.0 - a);

            a_ptr[i * cols + j] = a;
            b_ptr[i * cols + j] = b;
        }
    }

    return py::make_tuple(a_out, b_out);
}

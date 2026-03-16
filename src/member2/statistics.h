#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common.h"

namespace py = pybind11;

// ============================================================================
//  Histogram / PDF / CDF
// ============================================================================

// 256-bin histogram of a 2-D uint8 image.
inline npy<int> compute_histogram(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("compute_histogram: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int N = static_cast<int>(buf.shape[0] * buf.shape[1]);

    npy<int> result(256);
    auto rbuf = result.request();
    int* h = static_cast<int*>(rbuf.ptr);
    std::memset(h, 0, 256 * sizeof(int));

    for (int i = 0; i < N; ++i) h[ptr[i]]++;
    return result;
}

// PDF from a 256-bin histogram.
inline npy<double> compute_pdf(npy<int> hist) {
    py::buffer_info buf = hist.request();
    if (buf.ndim != 1 || buf.shape[0] != 256)
        throw std::runtime_error("compute_pdf: expected 256-element histogram.");
    const int* h = static_cast<const int*>(buf.ptr);

    long long total = 0;
    for (int i = 0; i < 256; ++i) total += h[i];

    npy<double> result(256);
    double* pdf = static_cast<double*>(result.request().ptr);
    double denom = (total > 0) ? static_cast<double>(total) : 1.0;
    for (int i = 0; i < 256; ++i) pdf[i] = h[i] / denom;
    return result;
}

// CDF from a 256-element PDF.
inline npy<double> compute_cdf(npy<double> pdf_arr) {
    py::buffer_info buf = pdf_arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 256)
        throw std::runtime_error("compute_cdf: expected 256-element PDF.");
    const double* pdf = static_cast<const double*>(buf.ptr);

    npy<double> result(256);
    double* cdf = static_cast<double*>(result.request().ptr);
    cdf[0] = pdf[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + pdf[i];
    return result;
}

// ============================================================================
//  Bi-Histogram Segmentation
// ============================================================================

// Segment the histogram at mean intensity I_m.
// Returns (I_m, hist_lower[0..I_m], hist_upper[I_m+1..255]).
inline py::tuple segment_histogram(npy<uint8_t> v) {
    py::buffer_info buf = v.request();
    if (buf.ndim != 2)
        throw std::runtime_error("segment_histogram: expected 2-D array.");
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    int N = static_cast<int>(buf.shape[0] * buf.shape[1]);

    // Mean intensity
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += ptr[i];
    int I_m = static_cast<int>(std::round(sum / N));
    I_m = clamp_idx(I_m, 0, 254);  // both sub-hists must have ≥1 bin

    // Full histogram
    int hist[256] = {};
    for (int i = 0; i < N; ++i) hist[ptr[i]]++;

    // Lower [0, I_m]
    int n_L = I_m + 1;
    npy<int> hist_lower(n_L);
    int* hl = static_cast<int*>(hist_lower.request().ptr);
    for (int i = 0; i < n_L; ++i) hl[i] = hist[i];

    // Upper [I_m+1, 255]
    int n_H = 255 - I_m;
    npy<int> hist_upper(n_H);
    int* hu = static_cast<int*>(hist_upper.request().ptr);
    for (int i = 0; i < n_H; ++i) hu[i] = hist[I_m + 1 + i];

    return py::make_tuple(I_m, hist_lower, hist_upper);
}

// ============================================================================
//  Adaptive Plateau Limits & Histogram Clipping
// ============================================================================
//
// For each sub-histogram (lower / upper):
//   T   = N_sub / n_bins                 (mean bin frequency)
//   E   = −Σ p_i log₂(p_i)              (discrete entropy)
//   Emax= log₂(n_bins)                  (maximum possible entropy)
//   PL  = max(1, T × E / Emax)          (adaptive plateau limit)
//
// Bins exceeding PL are clipped.  A new CDF is built for each sub-histogram.
// Returns a 256-element CDF array (lower CDF in [0..I_m], upper in [I_m+1..255]).
inline npy<double> compute_apl_and_clip(npy<int> hist_arr, int I_m) {
    py::buffer_info buf = hist_arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 256)
        throw std::runtime_error(
            "compute_apl_and_clip: expected 256-element histogram.");
    const int* hist_in = static_cast<const int*>(buf.ptr);

    I_m = clamp_idx(I_m, 0, 254);

    // Work on a floating-point copy
    std::vector<double> hist(256);
    for (int i = 0; i < 256; ++i) hist[i] = static_cast<double>(hist_in[i]);

    // --- Helper lambda: entropy of a contiguous sub-range ----------------
    auto sub_entropy = [&](int lo, int hi) -> double {
        double N_sub = 0.0;
        for (int i = lo; i <= hi; ++i) N_sub += hist[i];
        if (N_sub <= 0.0) return 0.0;

        double ent = 0.0;
        for (int i = lo; i <= hi; ++i) {
            if (hist[i] > 0.0) {
                double p = hist[i] / N_sub;
                ent -= p * std::log2(p);
            }
        }
        return ent;
    };

    // --- Lower sub-histogram [0, I_m] -----------------------------------
    int    n_L = I_m + 1;
    double N_L = 0.0;
    for (int i = 0; i <= I_m; ++i) N_L += hist[i];
    double T_L     = (n_L > 0 && N_L > 0) ? N_L / n_L : 0.0;
    double E_L     = sub_entropy(0, I_m);
    double Emax_L  = (n_L > 1) ? std::log2(static_cast<double>(n_L)) : 1.0;
    double PL_L    = std::max(1.0, T_L * E_L / (Emax_L + 1e-10));

    // --- Upper sub-histogram [I_m+1, 255] --------------------------------
    int    n_H = 255 - I_m;
    double N_H = 0.0;
    for (int i = I_m + 1; i <= 255; ++i) N_H += hist[i];
    double T_H     = (n_H > 0 && N_H > 0) ? N_H / n_H : 0.0;
    double E_H     = sub_entropy(I_m + 1, 255);
    double Emax_H  = (n_H > 1) ? std::log2(static_cast<double>(n_H)) : 1.0;
    double PL_H    = std::max(1.0, T_H * E_H / (Emax_H + 1e-10));

    // --- Clip ------------------------------------------------------------
    for (int i = 0; i <= I_m; ++i)
        if (hist[i] > PL_L) hist[i] = PL_L;
    for (int i = I_m + 1; i <= 255; ++i)
        if (hist[i] > PL_H) hist[i] = PL_H;

    // --- Rebuild PDF & CDF after clipping --------------------------------
    N_L = 0.0;
    for (int i = 0; i <= I_m; ++i) N_L += hist[i];
    N_H = 0.0;
    for (int i = I_m + 1; i <= 255; ++i) N_H += hist[i];

    npy<double> result(256);
    double* cdf = static_cast<double*>(result.request().ptr);
    std::memset(cdf, 0, 256 * sizeof(double));

    // Lower CDF [0, I_m]
    if (N_L > 0.0) {
        double cum = 0.0;
        for (int i = 0; i <= I_m; ++i) {
            cum += hist[i] / N_L;
            cdf[i] = cum;
        }
    }
    // Upper CDF [I_m+1, 255]
    if (N_H > 0.0) {
        double cum = 0.0;
        for (int i = I_m + 1; i <= 255; ++i) {
            cum += hist[i] / N_H;
            cdf[i] = cum;
        }
    }

    return result;
}

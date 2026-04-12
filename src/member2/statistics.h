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

    // Mean intensity — paper Eq.(10): Im = floor(sum(Ik * nk) / N)
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += ptr[i];
    int I_m = static_cast<int>(std::floor(sum / N));
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
// Paper Equations (12) and (13):
//   PLL = (H_hL / H_I) * (1/(I_m+1))     * sum_{k=0}^{I_m}   hL(k)
//   PLH = (H_hH / H_I) * (1/(L-I_m-1))  * sum_{k=I_m+1}^{L-1} hH(k)
//
// where H_I, H_hL, H_hH are discrete entropies of the full histogram,
// lower sub-histogram, and upper sub-histogram respectively (Eq.14).
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

    // --- Discrete entropy helper (Eq. 14): H = -sum p_i * log2(p_i) --------
    auto discrete_entropy = [&](int lo, int hi) -> double {
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

    // --- Entropy of full input histogram (H_I) -------------------------------
    double H_I = discrete_entropy(0, 255);
    if (H_I < 1e-10) H_I = 1e-10;  // guard against zero-entropy degenerate images

    // --- Lower sub-histogram [0, I_m] (paper Eq. 12) -----------------------
    //   Avg_L  = (1 / n_L) * sum hL(k)   where n_L = I_m + 1
    //   PL_L   = (H_hL / H_I) * Avg_L
    int    n_L = I_m + 1;
    double N_L = 0.0;
    for (int i = 0; i <= I_m; ++i) N_L += hist[i];
    double Avg_L = (n_L > 0) ? N_L / n_L : 0.0;
    double H_hL  = discrete_entropy(0, I_m);
    double PL_L  = (H_hL / H_I) * Avg_L;
    PL_L = std::max(1.0, PL_L);   // plateau must be ≥ 1

    // --- Upper sub-histogram [I_m+1, 255] (paper Eq. 13) -------------------
    //   n_H    = L - I_m - 1   (= 255 - I_m bins, since L=256)
    //   Avg_H  = (1 / n_H) * sum hH(k)
    //   PL_H   = (H_hH / H_I) * Avg_H
    int    n_H = 255 - I_m;  // number of bins in upper sub-histogram
    double N_H = 0.0;
    for (int i = I_m + 1; i <= 255; ++i) N_H += hist[i];
    double Avg_H = (n_H > 0) ? N_H / n_H : 0.0;
    double H_hH  = discrete_entropy(I_m + 1, 255);
    double PL_H  = (H_hH / H_I) * Avg_H;
    PL_H = std::max(1.0, PL_H);

    // --- Clip (paper Eq. 15-16) ---------------------------------------------
    for (int i = 0; i <= I_m; ++i)
        if (hist[i] > PL_L) hist[i] = PL_L;
    for (int i = I_m + 1; i <= 255; ++i)
        if (hist[i] > PL_H) hist[i] = PL_H;

    // --- Rebuild PDF & CDF after clipping (paper Eq. 17-18) ----------------
    // Recount after clipping
    N_L = 0.0;
    for (int i = 0; i <= I_m; ++i) N_L += hist[i];
    N_H = 0.0;
    for (int i = I_m + 1; i <= 255; ++i) N_H += hist[i];

    npy<double> result(256);
    double* cdf = static_cast<double*>(result.request().ptr);
    std::memset(cdf, 0, 256 * sizeof(double));

    // Lower CDF [0, I_m]: cdf(I_m) = 1 (paper Eq. 18)
    if (N_L > 0.0) {
        double cum = 0.0;
        for (int i = 0; i <= I_m; ++i) {
            cum += hist[i] / N_L;
            cdf[i] = cum;
        }
    }
    // Upper CDF [I_m+1, 255]: cdf(L-1) = 1 (paper Eq. 18)
    if (N_H > 0.0) {
        double cum = 0.0;
        for (int i = I_m + 1; i <= 255; ++i) {
            cum += hist[i] / N_H;
            cdf[i] = cum;
        }
    }

    return result;
}
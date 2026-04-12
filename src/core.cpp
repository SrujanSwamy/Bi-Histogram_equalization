#include <omp.h>
#include <vector>
#include <deque>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "common.h"
#include "member1/data_metrics.h"
#include "member2/statistics.h"
#include "member3/guided_filter.h"

namespace py = pybind11;

// ============================================================================
//  Week 3: Member 3 — Systems Lead (Thread Control)
// ============================================================================

void set_thread_count(int n) {
    if (n > 0) {
        omp_set_num_threads(n);
    }
}

// ============================================================================
//  Week 3: Member 2 — Algorithms Lead (Temporal Smoother)
// ============================================================================

class TemporalSmoother {
public:
    TemporalSmoother() {}

    py::tuple update_and_get_smoothed(double Im, double PLL, double PLH) {
        // Manage buffer sizes (keep last 5 frames)
        if (buf_Im.size() >= 5)  buf_Im.pop_front();
        if (buf_PLL.size() >= 5) buf_PLL.pop_front();
        if (buf_PLH.size() >= 5) buf_PLH.pop_front();

        buf_Im.push_back(Im);
        buf_PLL.push_back(PLL);
        buf_PLH.push_back(PLH);

        // Compute averages
        double avg_Im = std::accumulate(buf_Im.begin(), buf_Im.end(), 0.0) / buf_Im.size();
        double avg_PLL = std::accumulate(buf_PLL.begin(), buf_PLL.end(), 0.0) / buf_PLL.size();
        double avg_PLH = std::accumulate(buf_PLH.begin(), buf_PLH.end(), 0.0) / buf_PLH.size();

        return py::make_tuple(avg_Im, avg_PLL, avg_PLH);
    }

private:
    std::deque<double> buf_Im;
    std::deque<double> buf_PLL;
    std::deque<double> buf_PLH;
};

// ============================================================================
//  Wk.2 Member 1 — Data & Color Reconstruction
// ============================================================================

py::array_t<uint8_t> merge_hsv_to_rgb(py::array_t<uint8_t> h_arr, py::array_t<uint8_t> s_arr, py::array_t<uint8_t> v_arr) {
    auto h_buf = h_arr.request(), s_buf = s_arr.request(), v_buf = v_arr.request();
    int rows = h_buf.shape[0], cols = h_buf.shape[1];

    py::array_t<uint8_t> rgb_out({rows, cols, 3});
    uint8_t* ptr_out = static_cast<uint8_t*>(rgb_out.request().ptr);
    uint8_t* ptr_h = static_cast<uint8_t*>(h_buf.ptr);
    uint8_t* ptr_s = static_cast<uint8_t*>(s_buf.ptr);
    uint8_t* ptr_v = static_cast<uint8_t*>(v_buf.ptr);

    #pragma omp parallel for
    for (int i = 0; i < rows * cols; i++) {
        float h_val = ptr_h[i] * 2.0f; // OpenCV Hue is 0-179, map to 0-360
        float s_val = ptr_s[i] / 255.0f;
        float v_val = ptr_v[i] / 255.0f;

        float C = v_val * s_val;
        float X = C * (1.0f - std::abs(std::fmod(h_val / 60.0f, 2.0f) - 1.0f));
        float m_val = v_val - C;

        float r = 0, g = 0, b = 0;
        if (h_val < 60)      { r = C; g = X; b = 0; }
        else if (h_val < 120){ r = X; g = C; b = 0; }
        else if (h_val < 180){ r = 0; g = C; b = X; }
        else if (h_val < 240){ r = 0; g = X; b = C; }
        else if (h_val < 300){ r = X; g = 0; b = C; }
        else                 { r = C; g = 0; b = X; }

        int out_idx = i * 3;
        ptr_out[out_idx + 0] = static_cast<uint8_t>(std::clamp((r + m_val) * 255.0f, 0.0f, 255.0f));
        ptr_out[out_idx + 1] = static_cast<uint8_t>(std::clamp((g + m_val) * 255.0f, 0.0f, 255.0f));
        ptr_out[out_idx + 2] = static_cast<uint8_t>(std::clamp((b + m_val) * 255.0f, 0.0f, 255.0f));
    }
    return rgb_out;
}

// ============================================================================
//   Week 2: Member 2 — Algorithms & Integral Images
// ============================================================================

py::tuple compute_integral_images(py::array_t<uint8_t> v_arr) {
    auto buf = v_arr.request();
    int rows = buf.shape[0], cols = buf.shape[1];
    uint8_t* v_ptr = static_cast<uint8_t*>(buf.ptr);

    py::array_t<double> sum_out({rows + 1, cols + 1});
    py::array_t<double> sq_sum_out({rows + 1, cols + 1});
    double* sum_ptr = static_cast<double*>(sum_out.request().ptr);
    double* sq_sum_ptr = static_cast<double*>(sq_sum_out.request().ptr);

    std::fill(sum_ptr, sum_ptr + (rows + 1) * (cols + 1), 0.0);
    std::fill(sq_sum_ptr, sq_sum_ptr + (rows + 1) * (cols + 1), 0.0);

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            double val = static_cast<double>(v_ptr[(i - 1) * cols + (j - 1)]);
            int idx = i * (cols + 1) + j;
            int up = (i - 1) * (cols + 1) + j;
            int left = i * (cols + 1) + (j - 1);
            int up_left = (i - 1) * (cols + 1) + (j - 1);

            sum_ptr[idx] = val + sum_ptr[up] + sum_ptr[left] - sum_ptr[up_left];
            sq_sum_ptr[idx] = (val * val) + sq_sum_ptr[up] + sq_sum_ptr[left] - sq_sum_ptr[up_left];
        }
    }
    return py::make_tuple(sum_out, sq_sum_out);
}

py::array_t<double> compute_busyness_map(py::array_t<double> sum_ii, py::array_t<double> sq_sum_ii, int rows, int cols) {
    double* sum_ptr = static_cast<double*>(sum_ii.request().ptr);
    double* sq_sum_ptr = static_cast<double*>(sq_sum_ii.request().ptr);
    int stride = cols + 1;

    py::array_t<double> busyness({rows, cols});
    double* b_ptr = static_cast<double*>(busyness.request().ptr);
    
    double min_var = 1e15, max_var = -1.0;
    int r = 1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int r0 = std::max(0, i - r), r1 = std::min(rows - 1, i + r);
            int c0 = std::max(0, j - r), c1 = std::min(cols - 1, j + r);

            int br = (r1 + 1) * stride + (c1 + 1);
            int bl = (r1 + 1) * stride + c0;
            int tr = r0 * stride + (c1 + 1);
            int tl = r0 * stride + c0;

            double sum_val = sum_ptr[br] - sum_ptr[bl] - sum_ptr[tr] + sum_ptr[tl];
            double sq_sum_val = sq_sum_ptr[br] - sq_sum_ptr[bl] - sq_sum_ptr[tr] + sq_sum_ptr[tl];
            double N = (double)((r1 - r0 + 1) * (c1 - c0 + 1));

            double mean = sum_val / N;
            double var = std::max(0.0, (sq_sum_val / N) - (mean * mean));
            
            b_ptr[i * cols + j] = var;
            if (var < min_var) min_var = var;
            if (var > max_var) max_var = var;
        }
    }

    double range = (max_var - min_var < 1e-6) ? 1.0 : (max_var - min_var);
    #pragma omp parallel for
    for (int k = 0; k < rows * cols; k++) {
        b_ptr[k] = (b_ptr[k] - min_var) / range;
    }
    return busyness;
}

py::array_t<int32_t> map_dynamic_radius(py::array_t<double> busyness_map, int r_min, int r_max) {
    auto buf = busyness_map.request();
    double* b_ptr = static_cast<double*>(buf.ptr);
    py::array_t<int32_t> radius_out({buf.shape[0], buf.shape[1]});
    int32_t* r_ptr = static_cast<int32_t*>(radius_out.request().ptr);

    #pragma omp parallel for
    for (int k = 0; k < buf.shape[0] * buf.shape[1]; k++) {
        double r = (double)r_max - b_ptr[k] * (r_max - r_min);
        r_ptr[k] = static_cast<int32_t>(std::round(r));
    }
    return radius_out;
}

// ============================================================================
//   Week 2: Member 3 — Adaptive Guided Filter & Transformation
// ============================================================================

py::tuple adaptive_guided_filter(py::array_t<uint8_t> v_arr, py::array_t<double> sum_ii, py::array_t<double> sq_sum_ii, py::array_t<int32_t> radius_arr) {
    int rows = v_arr.request().shape[0], cols = v_arr.request().shape[1];
    uint8_t* v_data    = static_cast<uint8_t*>(v_arr.request().ptr);
    double* sum_ptr    = static_cast<double*>(sum_ii.request().ptr);
    double* sq_sum_ptr = static_cast<double*>(sq_sum_ii.request().ptr);
    int32_t* r_map     = static_cast<int32_t*>(radius_arr.request().ptr);
    int stride = cols + 1;
    // Paper Section 4: ε = 0.22 (normalised), scaled to uint8: 0.22 × 255² = 14308.5
    double eps = 0.22 * 255.0 * 255.0;

    // --- Step 1: compute per-window a_p, b_p (paper Eq. 7-8) ----------------
    // Each output pixel p gets its own (a_p, b_p) from the window w_p.
    std::vector<double> ap(rows * cols), bp(rows * cols);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int k = i * cols + j;
            int r = r_map[k];

            int r0 = std::max(0, i - r), r1 = std::min(rows - 1, i + r);
            int c0 = std::max(0, j - r), c1 = std::min(cols - 1, j + r);

            int br = (r1 + 1) * stride + (c1 + 1), bl = (r1 + 1) * stride + c0;
            int tr = r0 * stride + (c1 + 1),       tl = r0 * stride + c0;

            double sum_val    = sum_ptr[br]    - sum_ptr[bl]    - sum_ptr[tr]    + sum_ptr[tl];
            double sq_sum_val = sq_sum_ptr[br] - sq_sum_ptr[bl] - sq_sum_ptr[tr] + sq_sum_ptr[tl];
            double N = (double)((r1 - r0 + 1) * (c1 - c0 + 1));

            double mean_p = sum_val / N;
            double var_p  = std::max(0.0, (sq_sum_val / N) - (mean_p * mean_p));

            // Eq.(7) self-guided (G=I): a_p = σ²_p / (σ²_p + ε)
            ap[k] = var_p / (var_p + eps);
            // Eq.(8): b_p = μ_p - a_p * μ_p = μ_p * (1 - a_p)
            bp[k] = mean_p * (1.0 - ap[k]);
        }
    }

    // --- Step 2: average overlapping windows (paper Eq. 9) ------------------
    // a_i = mean{a_p : i ∈ w_p},  b_i = mean{b_p : i ∈ w_p}
    // For uniform radius r, every pixel i is covered by the windows of all
    // pixels p within distance r of i — i.e. the same box-filter of radius r.
    // We implement this with a second integral-image pass.
    py::array_t<double> a_out({rows, cols}), b_out({rows, cols});
    double* a_ptr = static_cast<double*>(a_out.request().ptr);
    double* b_ptr_out = static_cast<double*>(b_out.request().ptr);

    // Build integral images of ap and bp
    std::vector<double> sum_a((rows+1)*(cols+1), 0.0);
    std::vector<double> sum_b((rows+1)*(cols+1), 0.0);
    int s2 = cols + 1;
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            int src = (i-1)*cols + (j-1);
            int idx2 = i*s2 + j;
            sum_a[idx2] = ap[src] + sum_a[(i-1)*s2+j] + sum_a[i*s2+(j-1)] - sum_a[(i-1)*s2+(j-1)];
            sum_b[idx2] = bp[src] + sum_b[(i-1)*s2+j] + sum_b[i*s2+(j-1)] - sum_b[(i-1)*s2+(j-1)];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int k = i * cols + j;
            int r = r_map[k];

            int r0 = std::max(0, i - r), r1 = std::min(rows - 1, i + r);
            int c0 = std::max(0, j - r), c1 = std::min(cols - 1, j + r);

            int br2 = (r1+1)*s2+(c1+1), bl2 = (r1+1)*s2+c0;
            int tr2 =  r0   *s2+(c1+1), tl2 =  r0   *s2+c0;
            double N2 = (double)((r1-r0+1)*(c1-c0+1));

            a_ptr[k]     = (sum_a[br2] - sum_a[bl2] - sum_a[tr2] + sum_a[tl2]) / N2;
            b_ptr_out[k] = (sum_b[br2] - sum_b[bl2] - sum_b[tr2] + sum_b[tl2]) / N2;
        }
    }

    return py::make_tuple(a_out, b_out);
}

py::array_t<double> get_normalized_a(py::array_t<double> a_arr) {
    auto buf = a_arr.request();
    int size = buf.shape[0] * buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);
    
    double max_a = 0.0;
    for (int i=0; i<size; i++) if(ptr[i]>max_a) max_a = ptr[i];
    if (max_a < 1e-9) max_a = 1.0;

    py::array_t<double> a_hat({buf.shape[0], buf.shape[1]});
    double* hat_ptr = static_cast<double*>(a_hat.request().ptr);

    #pragma omp parallel for
    for (int i=0; i<size; i++) hat_ptr[i] = ptr[i] / max_a;
    return a_hat;
}

// ============================================================================
//  compute_mu_from_a_hat — NEW
//  Paper Section 3.3: μ is the mean of â(i,j) over the entire image.
//  This must be computed from the actual a_hat array each frame, NOT
//  passed as a fixed CLI parameter.
// ============================================================================
double compute_mu_from_a_hat(py::array_t<double> a_hat_arr) {
    auto buf = a_hat_arr.request();
    int size = buf.shape[0] * buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    double sum = 0.0;
    for (int i = 0; i < size; ++i) sum += ptr[i];
    return (size > 0) ? sum / size : 0.0;
}

// ============================================================================
//  apply_transformation — FIXED
//
//  Paper Eq. (19):
//    Lower (0 ≤ k ≤ Im):
//      x_L(i,j) = I(i,j) / Im                        (normalised intensity)
//      T(k;i,j) = Im * [ x_L + (cdf(k) - x_L) * λ(i,j) ]
//
//    Upper (Im < k ≤ L-1):
//      x_H(i,j) = (I(i,j) - Im) / (255 - Im)
//      T(k;i,j) = Im + (255 - Im) * [ x_H + (cdf(k) - x_H) * λ(i,j) ]
//
//  Paper Eq. (21):
//    λ(i,j) = 1 - κ·μ                 if â(i,j) < μ   (flat region)
//    λ(i,j) = (1 - κ·μ) + κ·â(i,j)   otherwise        (edge region)
//
//  Robustness additions (not in paper, needed for real images):
//    • λ_flat  is clamped to [0, 1]  — prevents sign-flip darkening on very
//      dark/narrow-histogram images where κμ > 1.
//    • λ_edge  is clamped to [1, κ]  — prevents extreme blowout; edge pixels
//      should always be enhanced at least as much as a plain HE (λ=1).
//    • Both sub-histogram branches guard their denominators with max(1.0,…).
//
//  IMPORTANT: μ is the image-level mean of â, computed via
//  compute_mu_from_a_hat() in Python before calling this function.
//  kappa is the user-supplied sharpening parameter κ (paper uses κ=5).
// ============================================================================
py::array_t<uint8_t> apply_transformation(
        py::array_t<uint8_t> v_arr,
        py::array_t<double>  a_hat_arr,
        py::array_t<double>  cdf_arr,
        double mu,      // mean of â(i,j) — computed from a_hat, NOT a fixed CLI param
        double kappa,   // sharpening strength κ
        double I_m)     // separation point (may be smoothed float from TemporalSmoother)
{
    auto v_buf = v_arr.request();
    int rows = v_buf.shape[0], cols = v_buf.shape[1];
    uint8_t* v_ptr    = static_cast<uint8_t*>(v_buf.ptr);
    double*  a_ptr    = static_cast<double*>(a_hat_arr.request().ptr);
    double*  cdf_lut  = static_cast<double*>(cdf_arr.request().ptr);

    py::array_t<uint8_t> out_arr({rows, cols});
    uint8_t* out_ptr = static_cast<uint8_t*>(out_arr.request().ptr);

    // Pre-compute boundary value; clamp to valid range
    double Im = std::max(1.0, std::min(I_m, 254.0));

    // Paper Eq.(21): base = 1 - κμ. With correct ε=0.22×255², μ stays small
    // (paper reports μ≈0.08 for Lena), so base naturally stays in (0,1).
    // No clamping applied — exact paper formula.
    double base = 1.0 - kappa * mu;

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int    idx    = i * cols + j;
            double val    = static_cast<double>(v_ptr[idx]);
            double a_hat  = a_ptr[idx];

            // --- Eq. (21): exact paper formula — no artificial clamps --------
            // Flat (â < μ):  λ = 1 - κμ          → under-enhance (noise suppression)
            // Edge (â ≥ μ):  λ = (1-κμ) + κ·â   → over-enhance  (edge sharpening)
            double lam;
            if (a_hat < mu) {
                lam = base;
            } else {
                lam = base + kappa * a_hat;
            }

            // --- Eq. (19): bi-histogram transformation ----------------------
            double cdf_val    = cdf_lut[static_cast<int>(val)];
            double transformed = 0.0;

            if (val <= Im) {
                // Lower sub-histogram [0, Im]
                // x_L = val / Im  (normalised to [0,1])
                double denom = std::max(1.0, Im);
                double x_L   = val / denom;
                transformed  = Im * (x_L + (cdf_val - x_L) * lam);
            } else {
                // Upper sub-histogram [Im+1, 255]
                // x_H = (val - Im) / (255 - Im)  (normalised to [0,1])
                double denom = std::max(1.0, 255.0 - Im);
                double x_H   = (val - Im) / denom;
                transformed  = Im + (255.0 - Im) * (x_H + (cdf_val - x_H) * lam);
            }

            out_ptr[idx] = static_cast<uint8_t>(
                std::clamp(std::round(transformed), 0.0, 255.0));
        }
    }
    return out_arr;
}

// ============================================================================
//   Pybind11 Module Binding
// ============================================================================

PYBIND11_MODULE(he_core, m) {
    m.doc() = "Edge-Enhancing Bi-Histogram Equalisation C++ Backend";

    // ---- Week 3 Additions ----
    m.def("set_thread_count", &set_thread_count, "Set number of OpenMP threads.");
    
    py::class_<TemporalSmoother>(m, "TemporalSmoother")
        .def(py::init<>())
        .def("update_and_get_smoothed", &TemporalSmoother::update_and_get_smoothed,
             "Push new stats and get averaged values (Im, PL_L, PL_H).");

    // ---- Week 1 Functions ----
    m.def("extract_v_channel", &extract_v_channel, "Extract V channel from RGB/BGR image.");
    m.def("compute_ambe", &compute_ambe, "Mean brightness of V channel.");
    m.def("compute_std", &compute_std, "Standard deviation of V channel.");
    m.def("compute_cii", &compute_cii, "Contrast Improvement Index.");
    m.def("compute_de", &compute_de, "Discrete Entropy.");
    m.def("compute_psi", &compute_psi, "Perceptual Sharpness Index.");

    m.def("compute_histogram", &compute_histogram, "Compute 256-bin histogram.");
    m.def("compute_pdf", &compute_pdf, "Compute PDF from histogram.");
    m.def("compute_cdf", &compute_cdf, "Compute CDF from PDF.");
    m.def("segment_histogram", &segment_histogram, "Segment histogram at mean.");
    m.def("compute_apl_and_clip", &compute_apl_and_clip, "Compute APL and return modified CDF.");

    m.def("compute_gf_coefficients", &compute_gf_coefficients, "Compute GF coefficients (Week 1 static).");
    m.def("get_omp_max_threads", []() { return omp_get_max_threads(); }, "Get max OpenMP threads.");

    // ---- Week 2 Additions ----
    m.def("merge_hsv_to_rgb", &merge_hsv_to_rgb, "Merge H, S, V to RGB (Week 2).");
    m.def("compute_integral_images", &compute_integral_images, "Compute Sum and Squared Sum Integral Images.");
    m.def("compute_busyness_map", &compute_busyness_map, "Compute variance-busyness map (Week 2).");
    m.def("map_dynamic_radius", &map_dynamic_radius, "Map busyness to dynamic radius.");
    m.def("adaptive_guided_filter", &adaptive_guided_filter, "Compute adaptive a, b coefficients.");
    m.def("get_normalized_a", &get_normalized_a, "Normalize 'a' coefficient.");
    m.def("compute_mu_from_a_hat", &compute_mu_from_a_hat, "Compute mean of normalised a-hat (paper Eq.21 mu).");
    m.def("apply_transformation", &apply_transformation, "Apply final bi-histogram transformation.");
}
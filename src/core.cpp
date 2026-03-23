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
    double* sum_ptr = static_cast<double*>(sum_ii.request().ptr);
    double* sq_sum_ptr = static_cast<double*>(sq_sum_ii.request().ptr);
    int32_t* r_map = static_cast<int32_t*>(radius_arr.request().ptr);
    int stride = cols + 1;
    double eps = 2000.0; 

    py::array_t<double> a_out({rows, cols}), b_out({rows, cols});
    double* a_ptr = static_cast<double*>(a_out.request().ptr);
    double* b_ptr = static_cast<double*>(b_out.request().ptr);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int k = i * cols + j;
            int r = r_map[k];
            
            int r0 = std::max(0, i - r), r1 = std::min(rows - 1, i + r);
            int c0 = std::max(0, j - r), c1 = std::min(cols - 1, j + r);

            int br = (r1 + 1) * stride + (c1 + 1), bl = (r1 + 1) * stride + c0;
            int tr = r0 * stride + (c1 + 1),       tl = r0 * stride + c0;

            double sum_val = sum_ptr[br] - sum_ptr[bl] - sum_ptr[tr] + sum_ptr[tl];
            double sq_sum_val = sq_sum_ptr[br] - sq_sum_ptr[bl] - sq_sum_ptr[tr] + sq_sum_ptr[tl];
            double N = (double)((r1 - r0 + 1) * (c1 - c0 + 1));

            double mean = sum_val / N;
            double var = std::max(0.0, (sq_sum_val / N) - (mean * mean));
            
            a_ptr[k] = var / (var + eps);
            b_ptr[k] = mean * (1.0 - a_ptr[k]);
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

py::array_t<uint8_t> apply_transformation(py::array_t<uint8_t> v_arr, py::array_t<double> a_hat_arr, py::array_t<double> cdf_arr, double mu, double kappa, double I_m) {
    auto v_buf = v_arr.request();
    int rows = v_buf.shape[0], cols = v_buf.shape[1];
    uint8_t* v_ptr = static_cast<uint8_t*>(v_buf.ptr);
    double* a_ptr = static_cast<double*>(a_hat_arr.request().ptr);
    double* cdf_lut = static_cast<double*>(cdf_arr.request().ptr); 

    py::array_t<uint8_t> out_arr({rows, cols});
    uint8_t* out_ptr = static_cast<uint8_t*>(out_arr.request().ptr);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            uint8_t val = v_ptr[idx];
            double a_hat = a_ptr[idx];
            
            double lam = (a_hat < mu) ? (1.0 - kappa * mu) : ((1.0 - kappa * mu) + kappa * a_hat);
            double cdf_val = cdf_lut[val]; 
            double transformed = 0.0;

            if ((double)val <= I_m) {
                double den = std::max(1e-5, I_m);
                double x_L = (double)val / den;
                transformed = I_m * (x_L + (cdf_val - x_L) * lam);
            } else {
                double den = std::max(1e-5, 255.0 - I_m);
                double x_H = ((double)val - I_m) / den;
                transformed = I_m + (255.0 - I_m) * (x_H + (cdf_val - x_H) * lam);
            }

            out_ptr[idx] = static_cast<uint8_t>(std::clamp(transformed, 0.0, 255.0));
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
    m.def("apply_transformation", &apply_transformation, "Apply final bi-histogram transformation.");
}
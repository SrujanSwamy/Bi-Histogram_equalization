// ============================================================================
//  Common types and utilities shared across all members
// ============================================================================
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Convenience type alias for pybind numpy arrays (force C-contiguous + type)
template <typename T>
using npy = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Clamp an index to [lo, hi]
static inline int clamp_idx(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

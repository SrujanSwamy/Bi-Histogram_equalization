# =============================================================================
# Edge-Enhancing Bi-Histogram Equalisation — Week 1 Build Configuration
# =============================================================================
#
# BUILD INSTRUCTIONS (from project root):
#
#   mkdir build && cd build
#   cmake .. -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
#   cmake --build . --config Release
#
# On Windows (PowerShell):
#   mkdir build; cd build
#   cmake .. -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
#   cmake --build . --config Release
#
# Prerequisites:
#   pip install pybind11 numpy opencv-python
#
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(he_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependencies ----------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ---- Module target ---------------------------------------------------------
pybind11_add_module(he_core src/core.cpp)
target_include_directories(he_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(he_core PRIVATE OpenMP::OpenMP_CXX)

# ---- Compiler optimizations ------------------------------------------------
if(MSVC)
    target_compile_options(he_core PRIVATE /O2 /W3)
else()
    target_compile_options(he_core PRIVATE -O2 -Wall -Wextra)
    # Statically link libgcc and libstdc++ to avoid runtime DLL issues on Windows (MinGW)
    if(MINGW)
        target_link_options(he_core PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

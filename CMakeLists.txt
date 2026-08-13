# Top-level CMake configuration for the libraries, tests, and macOS viewer.
#
# CMake uses separate configure and build steps. Run these commands from the
# repository root; each workflow gets its own build directory so their cached
# options do not interfere with one another.
#
# CPU tests (requires a C++23 compiler, Thrust from the bundled CCCL tree or a
# CUDA toolkit, and unless GTest is installed, network access during the first
# configure so FetchContent can download GoogleTest):
#   cmake -S . -B build/cpu-tests -DBUILD_TESTING=ON \
#     -DNBODY_ENABLE_CUDA=OFF -DNBODY_BUILD_VIEWER=OFF
#   cmake --build build/cpu-tests --parallel
#   ctest --test-dir build/cpu-tests --output-on-failure
#
# CUDA tests (requires the CUDA toolkit and a CUDA-capable compiler):
#   cmake -S . -B build/cuda-tests -DBUILD_TESTING=ON \
#     -DNBODY_ENABLE_CUDA=ON -DNBODY_BUILD_VIEWER=OFF
#   cmake --build build/cuda-tests --parallel
#   ctest --test-dir build/cuda-tests -L cuda --output-on-failure
#
# macOS Metal viewer:
#   cmake -S . -B build/viewer -DBUILD_TESTING=OFF \
#     -DNBODY_BUILD_VIEWER=ON
#   cmake --build build/viewer --target nbody_viewer
#   ./build/viewer/nbody_viewer -f data/test_spiral.bin
#
# Add "-G Ninja" to a configure command if Ninja is installed and desired.
# Change an option by re-running the configure command with -DOPTION=ON/OFF.
# Delete only the chosen build directory when you need a completely fresh
# configuration; source files must never be generated inside it manually.

# Version 3.28 provides the modern target and test features used below.
cmake_minimum_required(VERSION 3.28)

# Declaring only CXX here keeps CUDA and Objective-C++ optional. They are
# enabled later only when their corresponding build path is selected.
project(
    AMSC_Nbody
    VERSION 1.0.0
    DESCRIPTION "N-body simulation and CUDA tree algorithms"
    LANGUAGES CXX)

# CTest defines the standard BUILD_TESTING option (ON by default) and enables
# test registration. Pass -DBUILD_TESTING=OFF for a viewer-only build.
include(CTest)

# Apply the same portable C++23 baseline to ordinary C++ targets. Disabling
# extensions requests -std=c++23 rather than a compiler-specific GNU dialect.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The project uses conventional headers rather than C++ modules. Disable
# dependency scanning explicitly so C++23 targets also work with Makefile
# generators, including third-party targets created by FetchContent below.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Project-specific switches. Set them on the configure command line with, for
# example, -DNBODY_ENABLE_CUDA=ON. CUDA is opt-in; the viewer defaults to ON
# but is still built only on Apple platforms.
option(NBODY_ENABLE_CUDA "Build CUDA code and tests" OFF)
option(NBODY_BUILD_VIEWER "Build the macOS Metal viewer" ON)
option(
    NBODY_ENABLE_COMPUTE_SANITIZER_TESTS
    "Register Compute Sanitizer memcheck tests"
    OFF)

# The particle loader is header-only. nbody::particle gives consumers a stable,
# namespaced target and carries the project/Thrust include paths with it.
if(BUILD_TESTING)
    # Prefer the bundled CCCL package when it is present. Minimal server copies
    # may omit external/cccl, so fall back to the Thrust package belonging to
    # the CUDA toolkit selected by nvcc without enabling the CUDA language.
    set(
        nbody_bundled_thrust_dir
        "${PROJECT_SOURCE_DIR}/external/cccl/lib/cmake/thrust")
    if(EXISTS "${nbody_bundled_thrust_dir}/thrust-config.cmake")
        set(nbody_thrust_hints "${nbody_bundled_thrust_dir}")
    else()
        find_package(CUDAToolkit REQUIRED)
        file(
            GLOB nbody_toolkit_thrust_hints
            LIST_DIRECTORIES TRUE
            "${CUDAToolkit_LIBRARY_ROOT}/targets/*/lib/cmake/thrust")
        set(
            nbody_thrust_hints
            "${CUDAToolkit_TARGET_DIR}/lib/cmake/thrust"
            "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake/thrust"
            ${nbody_toolkit_thrust_hints})
    endif()

    find_package(Thrust REQUIRED CONFIG HINTS ${nbody_thrust_hints})
    thrust_create_target(nbody_thrust_cpp HOST CPP DEVICE CPP)

    add_library(nbody_particle INTERFACE)
    add_library(nbody::particle ALIAS nbody_particle)
    target_sources(
        nbody_particle
        INTERFACE
            FILE_SET HEADERS
            BASE_DIRS ${PROJECT_SOURCE_DIR}
            FILES src/particle.hpp)
    target_include_directories(
        nbody_particle
        INTERFACE
            ${PROJECT_SOURCE_DIR})
    target_link_libraries(nbody_particle INTERFACE nbody_thrust_cpp)
    target_compile_features(nbody_particle INTERFACE cxx_std_23)
endif()

# The Metal viewer is independent of the tests and CUDA library. Objective-C++
# and Apple frameworks are requested only when this target can actually build.
if(APPLE AND NBODY_BUILD_VIEWER)
    enable_language(OBJCXX)
    add_executable(nbody_viewer apps/viewer/main.mm)
    target_compile_features(nbody_viewer PRIVATE cxx_std_23)
    target_link_libraries(
        nbody_viewer
        PRIVATE
            "-framework Foundation"
            "-framework Cocoa"
            "-framework Metal"
            "-framework MetalKit"
            "-framework QuartzCore")
endif()

# Enabling CUDA creates the reusable nbody::cuda static library. Configuration
# stops with a clear error if the option is ON but nvcc is unavailable.
if(NBODY_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "NBODY_ENABLE_CUDA=ON, but no CUDA compiler was found")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Compile the production CUDA kernels once. CUDA test executables link this
    # library instead of recompiling the implementation sources themselves.
    add_library(
        nbody_cuda
        STATIC
        gpu/cuda_kernels/globalbounding.cu
        gpu/cuda_kernels/morton_leaf_groups.cu
        gpu/cuda_kernels/octree_builder.cu
        gpu/cuda_kernels/octree_physics.cu
        gpu/cuda_kernels/radix_to_octree.cu
        gpu/cuda_kernels/tree_builder.cu)
    add_library(nbody::cuda ALIAS nbody_cuda)

    # PUBLIC usage requirements are inherited by every target that links
    # nbody::cuda, including the include paths and CUDA runtime dependency.
    target_include_directories(
        nbody_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(nbody_cuda PUBLIC CUDA::cudart)
    # COMPILE_LANGUAGE generator expressions ensure these nvcc-specific flags
    # are used only while compiling CUDA sources. Line information makes CUDA
    # diagnostics and Compute Sanitizer reports point back to source lines.
    target_compile_options(
        nbody_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
    # CUDA currently uses C++17 even though host-only C++ uses C++23.
    # Separable compilation permits device code to call across translation
    # units; position-independent code keeps the static library reusable.
    set_target_properties(
        nbody_cuda
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS OFF
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
endif()

# Prefer an already installed GoogleTest package. If none is found, download
# the pinned release during configuration, then let tests/CMakeLists.txt define
# the actual test executables and their CTest entries.
if(BUILD_TESTING)
    find_package(GTest CONFIG QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_subdirectory(tests)
endif()

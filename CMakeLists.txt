# Top-level CMake configuration for the libraries, tests, and macOS viewer.
#
# CMake uses separate configure and build steps. Run these commands from the
# repository root; each workflow gets its own build directory so their cached
# options do not interfere with one another.
#
# CPU tests (requires a C++23 compiler and, during the first configure, network
# access so FetchContent can download the pinned CCCL and, unless GTest is
# installed, GoogleTest):
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
# CUDA simulation executable:
#   cmake -S . -B build/simulation -DBUILD_TESTING=OFF \
#     -DNBODY_ENABLE_CUDA=ON -DNBODY_BUILD_SIMULATION=ON \
#     -DNBODY_BUILD_VIEWER=OFF
#   cmake --build build/simulation --target nbody --parallel
#   ./build/simulation/nbody --help
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
option(NBODY_BUILD_SIMULATION "Build the CUDA simulation executable" OFF)
option(NBODY_BUILD_VIEWER "Build the macOS Metal viewer" ON)
option(NBODY_BUILD_STREAMING "Build the frame streaming server and client" ON)
option(
    NBODY_ENABLE_COMPUTE_SANITIZER_TESTS
    "Register Compute Sanitizer memcheck tests"
    OFF)

if(NBODY_BUILD_SIMULATION AND NOT NBODY_ENABLE_CUDA)
    message(
        FATAL_ERROR
        "NBODY_BUILD_SIMULATION=ON requires NBODY_ENABLE_CUDA=ON")
endif()

if(NBODY_BUILD_SIMULATION AND NOT NBODY_BUILD_STREAMING)
    message(
        FATAL_ERROR
        "NBODY_BUILD_SIMULATION=ON requires NBODY_BUILD_STREAMING=ON")
endif()

# Snapshot files and their transport are ordinary portable host code. Keeping
# this library independent of CUDA lets a laptop build only the download client
# while the cluster build links the same format implementation into nbody.
if(NBODY_BUILD_STREAMING)
    if(NOT UNIX)
        message(FATAL_ERROR "N-body streaming currently requires POSIX sockets")
    endif()

    add_library(
        nbody_streaming
        STATIC
        src/streaming/frame.cpp
        src/streaming/network.cpp
        src/streaming/server_process.cpp)
    add_library(nbody::streaming ALIAS nbody_streaming)
    target_include_directories(
        nbody_streaming
        PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(nbody_streaming PUBLIC cxx_std_17)
    target_compile_options(
        nbody_streaming
        PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(
        nbody_streaming
        PROPERTIES
            CXX_SCAN_FOR_MODULES OFF
            POSITION_INDEPENDENT_CODE ON)

    add_executable(nbody_stream_server apps/stream_server/main.cpp)
    target_link_libraries(nbody_stream_server PRIVATE nbody::streaming)
    target_compile_options(
        nbody_stream_server
        PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(
        nbody_stream_server PROPERTIES CXX_SCAN_FOR_MODULES OFF)

    add_executable(nbody_stream_client apps/stream_client/main.cpp)
    target_link_libraries(nbody_stream_client PRIVATE nbody::streaming)
    target_compile_options(
        nbody_stream_client
        PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(
        nbody_stream_client PROPERTIES CXX_SCAN_FOR_MODULES OFF)

    add_executable(nbody_browser_viewer apps/browser_viewer/main.cpp)
    target_link_libraries(nbody_browser_viewer PRIVATE nbody::streaming)
    target_compile_features(nbody_browser_viewer PRIVATE cxx_std_23)
    target_compile_options(
        nbody_browser_viewer
        PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(
        nbody_browser_viewer
        PRIVATE
            NBODY_BROWSER_ASSET_DIR="${PROJECT_SOURCE_DIR}/apps/browser_viewer/web")
    set_target_properties(
        nbody_browser_viewer PROPERTIES CXX_SCAN_FOR_MODULES OFF)
endif()

# The particle loader is header-only. nbody::particle gives consumers a stable,
# namespaced target and carries the project/Thrust include paths with it.
if(BUILD_TESTING)
    if(NBODY_ENABLE_CUDA)
        # CUDA builds must use the CCCL version shipped with their selected
        # toolkit. This keeps local-container and cluster compilation aligned
        # and avoids silently preferring an unrelated source-tree checkout.
        find_package(CUDAToolkit REQUIRED)
        file(
            GLOB nbody_toolkit_thrust_hints
            LIST_DIRECTORIES TRUE
            "${CUDAToolkit_LIBRARY_ROOT}/targets/*/lib/cmake/thrust")
        set(
            nbody_toolkit_thrust_hints
            "${CUDAToolkit_TARGET_DIR}/lib/cmake/thrust"
            "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake/thrust"
            ${nbody_toolkit_thrust_hints})
        find_package(
            Thrust
            REQUIRED
            CONFIG
            HINTS ${nbody_toolkit_thrust_hints}
            NO_DEFAULT_PATH)
    else()
        # Host-only macOS builds have no CUDA toolkit from which to obtain
        # Thrust. Fetch the exact CCCL commit validated with Apple Clang/LLVM 18
        # so CPU tests remain reproducible without keeping external/cccl in the
        # repository. CUDA builds never use this host-only fallback.
        include(FetchContent)
        FetchContent_Declare(
            nbody_cccl
            URL
                https://github.com/NVIDIA/cccl/archive/459e81a530f2b500a90ca24293c0c22a70cc89ef.tar.gz
            URL_HASH
                SHA256=428aa58a77d90259cd53090549fe6d8316e161ca22824f739c4b29d3051c5f15
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            EXCLUDE_FROM_ALL)
        FetchContent_MakeAvailable(nbody_cccl)
    endif()

    thrust_create_target(nbody_thrust_cpp HOST CPP DEVICE CPP)

    add_library(nbody_particle INTERFACE)
    add_library(nbody::particle ALIAS nbody_particle)
    target_sources(
        nbody_particle
        INTERFACE
            FILE_SET HEADERS
            BASE_DIRS ${PROJECT_SOURCE_DIR}
            FILES
                src/particle.hpp
                src/particle_type.hpp)
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
    target_include_directories(nbody_viewer PRIVATE ${PROJECT_SOURCE_DIR}/src)
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
        gpu/cuda_kernels/barnes_hut.cu
        gpu/cuda_kernels/globalbounding.cu
        gpu/cuda_kernels/integrator.cu
        gpu/cuda_kernels/morton.cu
        gpu/cuda_kernels/morton_leaf_groups.cu
        gpu/cuda_kernels/octree_builder.cu
        gpu/cuda_kernels/octree_geometry.cu
        gpu/cuda_kernels/octree_physics.cu
        gpu/cuda_kernels/radix_to_octree.cu
        gpu/cuda_kernels/simulation.cu
        gpu/cuda_kernels/tree_builder.cu
        src/simulation.cpp)
    add_library(nbody::cuda ALIAS nbody_cuda)

    # simulation.cpp contains host-side orchestration, but it includes the
    # CUDA-backed Particles type. Compile this source with the CUDA compiler so
    # Thrust consistently selects the CUDA device system in every translation
    # unit that contributes to nbody_cuda.
    set_source_files_properties(src/simulation.cpp PROPERTIES LANGUAGE CUDA)

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

    if(NBODY_BUILD_SIMULATION)
        add_executable(
            nbody
            apps/simulation/main.cpp
            src/streaming/device_capture.cu)
        # Particles owns Thrust device vectors, so compile the translation unit
        # with nvcc just like the host-side Simulation orchestration source.
        set_source_files_properties(
            apps/simulation/main.cpp
            PROPERTIES LANGUAGE CUDA)
        target_link_libraries(nbody PRIVATE nbody::cuda nbody::streaming)
        add_dependencies(nbody nbody_stream_server)
        target_compile_options(
            nbody
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
        set_target_properties(
            nbody
            PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_EXTENSIONS OFF)
    endif()
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

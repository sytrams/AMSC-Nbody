cmake_minimum_required(VERSION 3.28)

project(
    AMSC_Nbody
    VERSION 1.0.0
    DESCRIPTION "N-body simulation and CUDA tree algorithms"
    LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

option(NBODY_ENABLE_CUDA "Build CUDA code and tests" OFF)
option(NBODY_BUILD_VIEWER "Build the macOS Metal viewer" ON)
option(
    NBODY_ENABLE_COMPUTE_SANITIZER_TESTS
    "Register Compute Sanitizer memcheck tests"
    OFF)

if(BUILD_TESTING)
    add_library(nbody_particle)
    add_library(nbody::particle ALIAS nbody_particle)
    target_sources(
        nbody_particle
        PUBLIC
            FILE_SET CXX_MODULES
            FILES src/particle.cppm)
    target_compile_features(nbody_particle PUBLIC cxx_std_23)
    target_compile_options(
        nbody_particle
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>")
endif()

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

    target_include_directories(
        nbody_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(nbody_cuda PUBLIC CUDA::cudart)
    target_compile_options(
        nbody_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
    set_target_properties(
        nbody_cuda
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS OFF
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
endif()

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

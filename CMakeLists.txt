cmake_minimum_required(VERSION 3.28)

project(
    AMSC_Nbody
    VERSION 1.0.0
    DESCRIPTION "NVIDIA CUDA N-body simulation"
    LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(
        CMAKE_CUDA_ARCHITECTURES
        80
        CACHE STRING
        "CUDA architectures to compile for (80 targets NVIDIA A100)"
        FORCE)
endif()

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
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

add_library(nbody_particle STATIC src/particle.cpp)
add_library(nbody::particle ALIAS nbody_particle)

target_include_directories(
    nbody_particle
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src)

target_compile_features(nbody_particle PUBLIC cxx_std_20)
target_compile_options(
    nbody_particle
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

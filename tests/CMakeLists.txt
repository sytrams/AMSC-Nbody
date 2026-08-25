# This file is entered from the root CMakeLists.txt only when BUILD_TESTING is
# ON. Do not configure tests/ directly: configure the repository root so the
# nbody::particle and optional nbody::cuda targets already exist.
#
# After configuring and building, useful commands include:
#   ctest --test-dir build/cpu-tests --output-on-failure
#   ctest --test-dir build/cuda-tests -L cuda-unit --output-on-failure
#   ctest --test-dir build/cuda-tests -L cuda-integration --output-on-failure
#   ctest --test-dir build/cuda-tests -R TreeBuilder --output-on-failure
#   ctest --test-dir build/cuda-tests -N   # list tests without running them

# Provides gtest_discover_tests(), which asks each built GoogleTest executable
# for its cases and registers them individually with CTest.
include(GoogleTest)

# Host-only unit-test executable. GTest::gtest_main supplies main(), while the
# particle loader is consumed through its namespaced target alias.
add_executable(
    nbody_cpu_unit_tests
    unit/cpu/particle_file_test.cpp)
set_target_properties(
    nbody_cpu_unit_tests
    PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(
    nbody_cpu_unit_tests
    PRIVATE
        nbody::particle
        GTest::gtest_main)
# nbody::particle supplies Thrust's portable C++ host/device backend and the
# include paths for either the pinned host package or active-toolkit CCCL.
# PRE_TEST delays discovery until CTest runs. This is friendlier to cross builds
# because CMake does not need to execute the new binary during the build step.
# The prefix and label make CPU cases easy to select with -R and -L.
gtest_discover_tests(
    nbody_cpu_unit_tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "cpu."
    PROPERTIES
        LABELS "cpu-unit"
        TIMEOUT 30)

# The CUDA section is skipped cleanly when NBODY_ENABLE_CUDA=OFF. Checking the
# target is more robust than repeating the option because it verifies that the
# library was actually created.
if(TARGET nbody::cuda)
    # Define one CUDA GoogleTest executable and register all of its cases.
    # Usage inside this file:
    #   add_nbody_cuda_test_target(<target> <label> <source> [<source> ...])
    # ARGN is CMake's list of all source arguments after target and label.
    function(add_nbody_cuda_test_target target label)
        add_executable(${target} ${ARGN})
        # Test-only helpers stay PRIVATE; users of the production CUDA library
        # should not inherit paths under tests/.
        target_include_directories(
            ${target}
            PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
        target_link_libraries(
            ${target}
            PRIVATE
                nbody::cuda
                GTest::gtest_main)
        # Match the production CUDA diagnostics and language settings.
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
        set_target_properties(
            ${target}
            PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED YES
                CUDA_EXTENSIONS OFF
                CUDA_SEPARABLE_COMPILATION ON)
        # All cases in an executable share a label. RESOURCE_LOCK serializes
        # GPU test executables named "nbody_gpu" so concurrent CTest workers do
        # not contend for the same device. Run CTest with -L cuda-unit or
        # -L cuda-integration to select one group.
        gtest_discover_tests(
            ${target}
            DISCOVERY_MODE PRE_TEST
            TEST_PREFIX "cuda.${label}."
            PROPERTIES
                LABELS "cuda-${label}"
                RESOURCE_LOCK "nbody_gpu"
                TIMEOUT 300)
        # Remember each CUDA executable for the optional sanitizer loop below.
        set_property(
            GLOBAL APPEND PROPERTY NBODY_CUDA_TEST_TARGETS ${target})
    endfunction()

    # Focused tests of individual CUDA components.
    add_nbody_cuda_test_target(
        nbody_cuda_unit_tests
        unit
        unit/cuda/globalbounding_test.cu
        unit/cuda/morton_test.cu
        unit/cuda/morton_leaf_groups_test.cu
        unit/cuda/tree_builder_test.cu)

    # Tests that exercise several stages of the tree-building pipeline.
    add_nbody_cuda_test_target(
        nbody_cuda_integration_tests
        integration
        integration/cuda/radix_tree_groups_test.cu
        integration/cuda/radix_to_octree_test.cu
        integration/cuda/sparse_octree_test.cu
        integration/cuda/octree_physics_test.cu)

    # Opt-in because memcheck is much slower than the normal suite. Configure
    # with -DNBODY_ENABLE_COMPUTE_SANITIZER_TESTS=ON, build, then run:
    #   ctest --test-dir build/cuda-tests -L cuda-sanitizer --output-on-failure
    # Configuration fails if compute-sanitizer cannot be found in PATH or the
    # CUDA toolkit's bin directory.
    if(NBODY_ENABLE_COMPUTE_SANITIZER_TESTS)
        find_program(
            COMPUTE_SANITIZER_EXECUTABLE
            NAMES compute-sanitizer
            HINTS ${CUDAToolkit_BIN_DIR}
            REQUIRED)
        get_property(
            cuda_test_targets
            GLOBAL PROPERTY NBODY_CUDA_TEST_TARGETS)

        # Register one whole-executable memcheck per CUDA test target. These
        # entries also share the GPU resource lock and have a longer timeout.
        foreach(target IN LISTS cuda_test_targets)
            add_test(
                NAME "sanitizer.${target}"
                COMMAND
                    ${COMPUTE_SANITIZER_EXECUTABLE}
                    --tool memcheck
                    --leak-check full
                    $<TARGET_FILE:${target}>)
            set_tests_properties(
                "sanitizer.${target}"
                PROPERTIES
                    LABELS "cuda-sanitizer"
                    RESOURCE_LOCK "nbody_gpu"
                    TIMEOUT 900)
        endforeach()
    endif()
endif()

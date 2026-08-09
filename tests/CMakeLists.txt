include(GoogleTest)

add_executable(
    nbody_cpu_unit_tests
    unit/cpu/particle_file_test.cpp)
target_link_libraries(
    nbody_cpu_unit_tests
    PRIVATE
        nbody::particle
        GTest::gtest_main)
gtest_discover_tests(
    nbody_cpu_unit_tests
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "cpu."
    PROPERTIES
        LABELS "cpu-unit"
        TIMEOUT 30)

if(TARGET nbody::cuda)
    function(add_nbody_cuda_test_target target label)
        add_executable(${target} ${ARGN})
        target_include_directories(
            ${target}
            PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
        target_link_libraries(
            ${target}
            PRIVATE
                nbody::cuda
                GTest::gtest_main)
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
        gtest_discover_tests(
            ${target}
            DISCOVERY_MODE PRE_TEST
            TEST_PREFIX "cuda.${label}."
            PROPERTIES
                LABELS "cuda-${label}"
                RESOURCE_LOCK "nbody_gpu"
                TIMEOUT 300)
        set_property(
            GLOBAL APPEND PROPERTY NBODY_CUDA_TEST_TARGETS ${target})
    endfunction()

    add_nbody_cuda_test_target(
        nbody_cuda_unit_tests
        unit
        unit/cuda/morton_leaf_groups_test.cu
        unit/cuda/tree_builder_test.cu)

    add_nbody_cuda_test_target(
        nbody_cuda_integration_tests
        integration
        integration/cuda/radix_tree_groups_test.cu
        integration/cuda/radix_to_octree_test.cu
        integration/cuda/sparse_octree_test.cu
        integration/cuda/octree_physics_test.cu)

    if(NBODY_ENABLE_COMPUTE_SANITIZER_TESTS)
        find_program(
            COMPUTE_SANITIZER_EXECUTABLE
            NAMES compute-sanitizer
            HINTS ${CUDAToolkit_BIN_DIR}
            REQUIRED)
        get_property(
            cuda_test_targets
            GLOBAL PROPERTY NBODY_CUDA_TEST_TARGETS)

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

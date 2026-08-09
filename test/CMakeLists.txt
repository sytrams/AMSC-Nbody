find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

add_executable(particle_io_test particle_test.cpp)
target_link_libraries(
    particle_io_test
    PRIVATE
        nbody::particle
        GTest::gtest_main)

gtest_discover_tests(
    particle_io_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "particle.")

function(add_nbody_cuda_test target source)
    add_executable(${target} ${source})
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
            CUDA_SEPARABLE_COMPILATION ON)

    gtest_discover_tests(
        ${target}
        DISCOVERY_MODE PRE_TEST
        TEST_PREFIX "cuda."
        PROPERTIES
            LABELS "gpu"
            RESOURCE_LOCK "nbody_gpu"
            TIMEOUT 300)

    set_property(GLOBAL APPEND PROPERTY NBODY_CUDA_TEST_TARGETS ${target})
endfunction()

add_nbody_cuda_test(
    morton_leaf_groups_test
    ${PROJECT_SOURCE_DIR}/test_morton_groups/morton_leaf_groups_test.cu)
add_nbody_cuda_test(
    tree_builder_test
    ${PROJECT_SOURCE_DIR}/test_tree/tree_builder_test.cu)
add_nbody_cuda_test(
    radix_tree_unique_keys_test
    ${PROJECT_SOURCE_DIR}/test_radix_groups/radix_tree_unique_keys_test.cu)
add_nbody_cuda_test(
    radix_to_octree_plan_test
    ${PROJECT_SOURCE_DIR}/test_octree_plan/radix_to_octree_plan_test.cu)
add_nbody_cuda_test(
    sparse_octree_test
    ${PROJECT_SOURCE_DIR}/test_octree/sparse_octree_test.cu)
add_nbody_cuda_test(
    octree_mass_com_test
    ${PROJECT_SOURCE_DIR}/test_octree_physics/octree_mass_com_test.cu)

option(
    NBODY_ENABLE_COMPUTE_SANITIZER_TESTS
    "Register Compute Sanitizer memcheck tests"
    OFF)

if(NBODY_ENABLE_COMPUTE_SANITIZER_TESTS)
    find_program(
        COMPUTE_SANITIZER_EXECUTABLE
        NAMES compute-sanitizer
        HINTS ${CUDAToolkit_BIN_DIR}
        REQUIRED)

    get_property(cuda_test_targets GLOBAL PROPERTY NBODY_CUDA_TEST_TARGETS)
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
                LABELS "gpu;sanitizer"
                RESOURCE_LOCK "nbody_gpu"
                TIMEOUT 900)
    endforeach()
endif()

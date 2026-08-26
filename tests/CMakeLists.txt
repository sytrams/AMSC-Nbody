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

# The data tools depend on NumPy. Register their standard-library unittests
# when both Python and that optional runtime dependency are available.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE nbody_numpy_import_status
        OUTPUT_QUIET
        ERROR_QUIET)
    if(nbody_numpy_import_status EQUAL 0)
        add_test(
            NAME "cpu.GenerateGalaxyTest"
            COMMAND
                ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/tests/unit/python/generate_galaxy_test.py")
        add_test(
            NAME "cpu.DownloadSolarSystemTest"
            COMMAND
                ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/tests/unit/python/download_solar_system_test.py")
        set_tests_properties(
            "cpu.GenerateGalaxyTest"
            "cpu.DownloadSolarSystemTest"
            PROPERTIES
                LABELS "cpu-unit"
                TIMEOUT 30)
    else()
        message(
            STATUS
            "NumPy is unavailable; the generate_galaxy.py test is not registered")
    endif()
endif()

# The CUDA section is skipped cleanly when NBODY_ENABLE_CUDA=OFF. Checking the
# target is more robust than repeating the option because it verifies that the
# library was actually created.
if(TARGET nbody::cuda)
    # WSL exposes the Windows NVIDIA driver through /usr/lib/wsl/lib. The CUDA
    # runtime may otherwise resolve a Linux stub libcuda first, causing every
    # GPU test to report that no CUDA-capable device is available. Apply this
    # only when the WSL driver library is actually present, so native Linux and
    # cluster environments remain unchanged.
    set(NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION "")
    if(
        CMAKE_SYSTEM_NAME STREQUAL "Linux"
        AND EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
        set(
            NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:/usr/lib/wsl/lib")
        message(
            STATUS
            "WSL CUDA driver detected: CTest will prepend /usr/lib/wsl/lib to LD_LIBRARY_PATH")
    endif()

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
        set(
            cuda_test_properties
            LABELS "cuda-${label}"
            RESOURCE_LOCK "nbody_gpu"
            TIMEOUT 300)
        if(NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION)
            list(
                APPEND cuda_test_properties
                ENVIRONMENT_MODIFICATION
                "${NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION}")
        endif()

        gtest_discover_tests(
            ${target}
            DISCOVERY_MODE PRE_TEST
            TEST_PREFIX "cuda.${label}."
            PROPERTIES ${cuda_test_properties})
        # Remember each CUDA executable for the optional sanitizer loop below.
        set_property(
            GLOBAL APPEND PROPERTY NBODY_CUDA_TEST_TARGETS ${target})
    endfunction()

    # Focused tests of individual CUDA components.
    add_nbody_cuda_test_target(
        nbody_cuda_unit_tests
        unit
        unit/cuda/barnes_hut_test.cu
        unit/cuda/globalbounding_test.cu
        unit/cuda/integrator_test.cu
        unit/cuda/morton_test.cu
        unit/cuda/morton_leaf_groups_test.cu
        unit/cuda/octree_geometry_test.cu
        unit/cuda/tree_builder_test.cu)

    # Tests that exercise several stages of the tree-building pipeline.
    add_nbody_cuda_test_target(
        nbody_cuda_integration_tests
        integration
        integration/cuda/radix_tree_groups_test.cu
        integration/cuda/radix_to_octree_test.cu
        integration/cuda/sparse_octree_test.cu
        integration/cuda/octree_physics_test.cu
        integration/cuda/particle_octree_pipeline_test.cu
        integration/cuda/simulation_test.cu
        integration/cuda/two_body_orbit_test.cu)

    # Exercise the production executable as a separate process. The small
    # helper writes a deterministic binary input without relying on Python or
    # checking generated binary data into the repository.
    if(TARGET nbody)
        add_executable(
            nbody_cli_fixture_writer
            support/write_particle_fixture.cpp)
        set_target_properties(
            nbody_cli_fixture_writer
            PROPERTIES CXX_SCAN_FOR_MODULES OFF)

        add_test(
            NAME "cuda.integration.SimulationCliBehavior"
            COMMAND
                ${CMAKE_COMMAND}
                "-DNBODY_CLI=$<TARGET_FILE:nbody>"
                "-DNBODY_FIXTURE_WRITER=$<TARGET_FILE:nbody_cli_fixture_writer>"
                "-DNBODY_TEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/data"
                -P
                "${PROJECT_SOURCE_DIR}/tests/scripts/simulation_cli_test.cmake")
        set(
            simulation_cli_test_properties
            LABELS "cuda-integration"
            RESOURCE_LOCK "nbody_gpu"
            TIMEOUT 300)
        if(NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION)
            list(
                APPEND simulation_cli_test_properties
                ENVIRONMENT_MODIFICATION
                "${NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION}")
        endif()
        set_tests_properties(
            "cuda.integration.SimulationCliBehavior"
            PROPERTIES ${simulation_cli_test_properties})
    endif()

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
                    --error-exitcode 1
                    --tool memcheck
                    --leak-check full
                    $<TARGET_FILE:${target}>)
            set(
                sanitizer_test_properties
                LABELS "cuda-sanitizer"
                RESOURCE_LOCK "nbody_gpu"
                TIMEOUT 900)
            if(NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION)
                list(
                    APPEND sanitizer_test_properties
                    ENVIRONMENT_MODIFICATION
                    "${NBODY_CUDA_TEST_ENVIRONMENT_MODIFICATION}")
            endif()
            set_tests_properties(
                "sanitizer.${target}"
                PROPERTIES ${sanitizer_test_properties})
        endforeach()
    endif()
endif()

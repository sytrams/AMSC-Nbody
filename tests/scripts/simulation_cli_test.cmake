foreach(required_variable IN ITEMS NBODY_CLI NBODY_FIXTURE_WRITER NBODY_TEST_DIRECTORY)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} must be provided")
    endif()
endforeach()

function(expect_text text expected context)
    string(FIND "${text}" "${expected}" match_position)
    if(match_position EQUAL -1)
        message(
            FATAL_ERROR
            "${context}: expected output to contain '${expected}'\nActual output:\n${text}")
    endif()
endfunction()

function(expect_status actual expected context)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(
            FATAL_ERROR
            "${context}: expected exit status ${expected}, got ${actual}")
    endif()
endfunction()

file(MAKE_DIRECTORY "${NBODY_TEST_DIRECTORY}")
set(particle_file "${NBODY_TEST_DIRECTORY}/simulation-cli-particle.bin")
set(missing_file "${NBODY_TEST_DIRECTORY}/simulation-cli-missing.bin")
file(REMOVE "${particle_file}" "${missing_file}")

execute_process(
    COMMAND "${NBODY_CLI}" --help
    RESULT_VARIABLE help_status
    OUTPUT_VARIABLE help_output
    ERROR_VARIABLE help_error)
expect_status("${help_status}" 0 "--help")
expect_text("${help_output}" "Usage:" "--help")
expect_text("${help_output}" "--time-step DT" "--help")
expect_text("${help_output}" "--softening VALUE" "--help")

execute_process(
    COMMAND "${NBODY_CLI}"
    RESULT_VARIABLE empty_status
    OUTPUT_VARIABLE empty_output
    ERROR_VARIABLE empty_error)
expect_status("${empty_status}" 2 "empty command line")
expect_text("${empty_error}" "--input is required" "empty command line")

execute_process(
    COMMAND "${NBODY_CLI}" --unknown
    RESULT_VARIABLE unknown_status
    OUTPUT_VARIABLE unknown_output
    ERROR_VARIABLE unknown_error)
expect_status("${unknown_status}" 2 "unknown option")
expect_text("${unknown_error}" "Unknown option: --unknown" "unknown option")

execute_process(
    COMMAND
        "${NBODY_CLI}"
        --input ignored.bin
        --time-step 0.25
        --steps 0
    RESULT_VARIABLE invalid_steps_status
    OUTPUT_VARIABLE invalid_steps_output
    ERROR_VARIABLE invalid_steps_error)
expect_status("${invalid_steps_status}" 2 "zero steps")
expect_text("${invalid_steps_error}" "Invalid positive integer for --steps"
            "zero steps")

execute_process(
    COMMAND
        "${NBODY_CLI}"
        --input "${missing_file}"
        --time-step 0.25
        --steps 2
    RESULT_VARIABLE missing_file_status
    OUTPUT_VARIABLE missing_file_output
    ERROR_VARIABLE missing_file_error)
expect_status("${missing_file_status}" 1 "missing input file")
expect_text("${missing_file_error}" "Could not open particle input file"
            "missing input file")

execute_process(
    COMMAND "${NBODY_FIXTURE_WRITER}" "${particle_file}"
    RESULT_VARIABLE fixture_status
    OUTPUT_VARIABLE fixture_output
    ERROR_VARIABLE fixture_error)
expect_status("${fixture_status}" 0 "particle fixture writer")

execute_process(
    COMMAND
        "${NBODY_CLI}"
        --input "${particle_file}"
        --time-step 0.25
        --steps 2
        --theta 0.6
        --softening 0.03
    RESULT_VARIABLE simulation_status
    OUTPUT_VARIABLE simulation_output
    ERROR_VARIABLE simulation_error)
file(REMOVE "${particle_file}")

expect_status("${simulation_status}" 0 "valid simulation")
expect_text("${simulation_output}" "Loaded 1 particles" "valid simulation")
expect_text("${simulation_output}" "Simulation complete" "valid simulation")
expect_text("${simulation_output}" "steps: 2" "valid simulation")
expect_text("${simulation_output}" "simulated time: 0.5" "valid simulation")

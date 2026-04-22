cmake_minimum_required(VERSION 3.31)

if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@18/bin/clang++" CACHE FILEPATH "" FORCE)
endif()

project(VectorTesting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Scarica Google Test automaticamente
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip

)
# Impedisce a GTest di sovrascrivere le impostazioni del compilatore su Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(run_tests)
target_sources(run_tests
  PRIVATE
    main.cpp
  PRIVATE
    FILE_SET CXX_MODULES
    FILES
      ../src/vector.cppm
)

# Collega le librerie di GTest (include e link automatici)
target_link_libraries(run_tests PRIVATE GTest::gtest_main)

if(APPLE)
  target_compile_options(run_tests PRIVATE -stdlib=libc++)
  target_link_options(run_tests PRIVATE -stdlib=libc++)
endif()

# Permette di usare 'ctest' per lanciare i test
include(GoogleTest)
gtest_discover_tests(run_tests)

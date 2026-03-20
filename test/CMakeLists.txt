cmake_minimum_required(VERSION 3.14)
project(VectorTesting)

# Imposta lo standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Scarica Google Test automaticamente
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip

)
# Impedisce a GTest di sovrascrivere le impostazioni del compilatore su Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Definisci il tuo eseguibile di test
add_executable(run_tests main.cpp)

# Collega le librerie di GTest (include e link automatici)
target_link_libraries(run_tests GTest::gtest_main)

# Permette di usare 'ctest' per lanciare i test
include(GoogleTest)
gtest_discover_tests(run_tests)

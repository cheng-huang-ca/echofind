cmake_minimum_required(VERSION 3.16)
project(echofind_edge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(EF_PYBIND "Build the pybind11 module (needs pybind11 and Python headers)" OFF)
option(EF_NATIVE "Tune for the build machine (-march=native); off for portable binaries" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -O2)
  # complex multiply without the C99 Annex G NaN/Inf recovery (__muldc3 calls): the inputs are
  # finite sonar samples, and the golden and parity tests confirm results are unchanged
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fcx-limited-range EF_HAS_CX_LIMITED)
  if(EF_HAS_CX_LIMITED)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcx-limited-range>)
  endif()
  if(EF_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# Shared library: DSP C API (for ctypes parity tests) plus the generated classifiers.
add_library(ef_edge SHARED src/capi.cpp generated/models.c)
target_include_directories(ef_edge PUBLIC include)
if(MINGW)
  target_link_options(ef_edge PRIVATE -static-libgcc -static-libstdc++ -static)
endif()

# Benchmark and CLI: one ping through the whole chain, timed per stage.
add_executable(ef_bench bench/ef_bench.cpp generated/models.c)
target_include_directories(ef_bench PRIVATE include)
if(WIN32)
  target_link_libraries(ef_bench PRIVATE psapi)
endif()
if(MINGW)
  target_link_options(ef_bench PRIVATE -static)
endif()

# C++-only parity test against golden vectors exported from Python (tests/golden/*.bin).
add_executable(ef_golden tests/golden_test.cpp)
target_include_directories(ef_golden PRIVATE include)
if(MINGW)
  target_link_options(ef_golden PRIVATE -static)
endif()
enable_testing()
add_test(NAME golden COMMAND ef_golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden)

if(EF_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_ef_edge bindings/pybind.cpp generated/models.c)
  target_include_directories(_ef_edge PRIVATE include)
endif()

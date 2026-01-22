cmake_minimum_required(VERSION 3.15)
project(matmul_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_AVX2 "Enable AVX2/FMA" ON)

add_executable(matmul_bench
  src/main.cpp
  src/matrix.cpp
  src/gemm_ref.cpp
  src/gemm_base.cpp
  src/gemm_opt_32.cpp
  src/gemm_opt_4096.cpp
)

target_include_directories(matmul_bench PRIVATE include)

target_compile_definitions(matmul_bench PRIVATE $<$<BOOL:${ENABLE_AVX2}>:MATMUL_ENABLE_AVX2>)

if(MSVC)
  target_compile_options(matmul_bench PRIVATE /O2 /fp:fast)
  if(ENABLE_AVX2)
    target_compile_options(matmul_bench PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(matmul_bench PRIVATE -O3 -ffast-math -fno-rtti)
  if(ENABLE_AVX2)
    target_compile_options(matmul_bench PRIVATE -mavx2 -mfma)
  endif()
endif()

cmake_minimum_required(VERSION 3.18)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# CUDA is enabled on demand below (only if we find cuDSS headers).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Resolve the JAX FFI header directory.
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_FFI_DIR)
message(STATUS "JAX FFI include directory: ${XLA_FFI_DIR}")

find_package(nanobind CONFIG REQUIRED)

# cuDSS ships via the `nvidia-cudss-cu13` wheel alongside the CUDA runtime
# headers/libraries. Let the user point us at an install root; otherwise probe
# the active Python environment for the nvidia/cu13 layout that the wheel uses.
if(NOT DEFINED SPARSEJAX_CUDSS_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import importlib.util as u, sys; s = u.find_spec('nvidia.cu13'); \
       print(list(s.submodule_search_locations)[0] if s and s.submodule_search_locations else '')"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE SPARSEJAX_CUDSS_ROOT)
endif()

if(SPARSEJAX_CUDSS_ROOT AND EXISTS "${SPARSEJAX_CUDSS_ROOT}/include/cudss.h")
  message(STATUS "cuDSS root: ${SPARSEJAX_CUDSS_ROOT}")
  set(CUDSS_INCLUDE_DIR "${SPARSEJAX_CUDSS_ROOT}/include")
  set(CUDSS_LIB_DIR "${SPARSEJAX_CUDSS_ROOT}/lib")
  set(SPARSEJAX_HAVE_CUDSS ON)
else()
  message(WARNING
    "cuDSS headers not found. Set -DSPARSEJAX_CUDSS_ROOT=<path> or "
    "install nvidia-cudss-cu13. Skipping native FFI extension.")
  set(SPARSEJAX_HAVE_CUDSS OFF)
endif()

if(SPARSEJAX_HAVE_CUDSS)
  # crt/host_config.h and other CUDA toolkit headers are not shipped in the
  # minimal cuDSS wheel; fall back to a system CUDA install if present.
  set(SPARSEJAX_CUDA_SYSTEM_DIR "")
  set(SPARSEJAX_CUDA_ROOT "")
  foreach(_dir /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.2 /usr/local/cuda-13.1)
    if(EXISTS "${_dir}/include/crt/host_config.h")
      set(SPARSEJAX_CUDA_SYSTEM_DIR "${_dir}/include")
      set(SPARSEJAX_CUDA_ROOT "${_dir}")
      break()
    endif()
  endforeach()
  if(SPARSEJAX_CUDA_SYSTEM_DIR)
    message(STATUS "System CUDA includes: ${SPARSEJAX_CUDA_SYSTEM_DIR}")
  endif()

  # Point CMake at nvcc from the discovered CUDA install when it isn't on PATH.
  if(SPARSEJAX_CUDA_ROOT AND EXISTS "${SPARSEJAX_CUDA_ROOT}/bin/nvcc"
     AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${SPARSEJAX_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "")
  endif()

  # Default to a broad arch list before enable_language so it sticks.
  # CMake's CUDA detection otherwise initializes CMAKE_CUDA_ARCHITECTURES to
  # the compiler's default (often a single SM that's older than the user's
  # GPU), which leaves us shipping PTX that nvrtc declines to JIT cleanly on
  # newer hardware. Users can still override with -DCMAKE_CUDA_ARCHITECTURES.
  if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
        "CUDA architectures to compile for")
  endif()

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  nanobind_add_module(backend MODULE
    csrc/module.cc
    csrc/cudss_logdet_kernel.cu
  )
  target_include_directories(backend PRIVATE
    ${XLA_FFI_DIR}
    ${CUDSS_INCLUDE_DIR}
    ${SPARSEJAX_CUDA_SYSTEM_DIR}
  )
  target_link_directories(backend PRIVATE ${CUDSS_LIB_DIR})
  target_link_libraries(backend PRIVATE cudss)
  # The cuDSS wheel lib directory is not on rpath by default; set it so the
  # extension can find libcudss.so.0 without extra LD_LIBRARY_PATH fiddling.
  set_target_properties(backend PROPERTIES
    INSTALL_RPATH "${CUDSS_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  install(TARGETS backend DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

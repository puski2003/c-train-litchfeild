cmake_minimum_required(VERSION 3.30)

set(RAW_VCPKG_PATH $ENV{VCPKG_ROOT})
file(TO_CMAKE_PATH "${RAW_VCPKG_PATH}" VCPKG_ROOT)
if("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "" AND NOT "${VCPKG_ROOT}" STREQUAL "")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

project(LichtFeldTrainColmapHeadless VERSION 0.1.0 LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external")
    set(LFS_REPO_ROOT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    set(LFS_REPO_ROOT_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()
set(LFS_REPO_ROOT "${LFS_REPO_ROOT_DEFAULT}" CACHE PATH "Path to the standalone repo root or LichtFeld-Studio repository root")
get_filename_component(LFS_REPO_ROOT "${LFS_REPO_ROOT}" ABSOLUTE)

function(lfs_ensure_repo_link name)
    set(link_path "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    set(real_path "${LFS_REPO_ROOT}/${name}")
    if(EXISTS "${link_path}")
        return()
    endif()
    if(NOT EXISTS "${real_path}")
        message(FATAL_ERROR "Expected repo path does not exist: ${real_path}")
    endif()

    file(CREATE_LINK "${real_path}" "${link_path}" SYMBOLIC RESULT link_result)
    if(NOT "${link_result}" STREQUAL "0")
        message(FATAL_ERROR
            "Could not create ${link_path} -> ${real_path}.\n"
            "This standalone project reuses upstream CMake files that expect CMAKE_SOURCE_DIR/src "
            "and CMAKE_SOURCE_DIR/external to exist. On Linux/Colab this should be a normal symlink. "
            "Error: ${link_result}")
    endif()
endfunction()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    # Upstream module CMake files refer to ${CMAKE_SOURCE_DIR}/src and /external.
    # These links let this folder behave like a tiny source root without copying code.
    lfs_ensure_repo_link(src)
    lfs_ensure_repo_link(external)
    lfs_ensure_repo_link(cmake)
endif()

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${LFS_REPO_ROOT}"
    OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git describe --tags --dirty --always --match "v*"
    WORKING_DIRECTORY "${LFS_REPO_ROOT}"
    OUTPUT_VARIABLE GIT_TAGGED_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH_SHORT)
    set(GIT_COMMIT_HASH_SHORT "unknown")
endif()
if(NOT GIT_TAGGED_VERSION)
    set(GIT_TAGGED_VERSION "${GIT_COMMIT_HASH_SHORT}")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
file(WRITE "${CMAKE_BINARY_DIR}/include/git_version.h"
"#pragma once
#define GIT_COMMIT_HASH_SHORT \"${GIT_COMMIT_HASH_SHORT}\"
#define GIT_TAGGED_VERSION \"${GIT_TAGGED_VERSION}\"
")

set(PROJECT_ROOT_PATH "${LFS_REPO_ROOT}")
set(PROJECT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(PROJECT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(PROJECT_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(PROJECT_VERSION "${PROJECT_VERSION}")
configure_file(
    "${LFS_REPO_ROOT}/cmake/config.h.in"
    "${CMAKE_BINARY_DIR}/include/config.h"
    @ONLY
)
add_compile_definitions(PROJECT_ROOT_PATH="${LFS_REPO_ROOT}")

option(BUILD_CUDA_PTX_ONLY "Build PTX-only for smaller binaries and runtime JIT" OFF)
set(BUILD_CUDA_MIN_SM "75" CACHE STRING "Minimum SM for PTX-only builds")
option(ENABLE_ALLOCATION_PROFILING "Enable tensor allocation profiling" OFF)
option(ENABLE_TENSOR_VALIDATION "Enable runtime tensor validation" OFF)
option(ENABLE_CUDA_DEBUG_SYNC "Enable synchronous CUDA kernel checking" OFF)
option(ENABLE_TENSOR_OP_TRACING "Enable tensor operation tracing" OFF)
option(CUDA_DEVICE_DEBUG "Enable CUDA device debug flags in Debug builds" OFF)
option(BUILD_PORTABLE "Compatibility option consumed by upstream install blocks" OFF)

find_program(NVIDIA_SMI nvidia-smi)
if(NVIDIA_SMI)
    execute_process(
        COMMAND "${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REGEX REPLACE "\n.*" "" DETECTED_COMPUTE_CAP "${DETECTED_COMPUTE_CAP}")
else()
    set(DETECTED_COMPUTE_CAP "8.6")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)
if(BUILD_CUDA_PTX_ONLY)
    set(LichtFeld-Studio_CUDA_ARCH "${BUILD_CUDA_MIN_SM}-virtual")
else()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
        string(REPLACE "." "" LichtFeld-Studio_CUDA_ARCH "${DETECTED_COMPUTE_CAP}")
    else()
        set(LichtFeld-Studio_CUDA_ARCH "native")
    endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}" CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
set(LFS_CUDA_DEVICE_DEBUG_FLAGS "")

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(args REQUIRED)
find_package(spdlog REQUIRED)
find_package(WebP REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(pxr CONFIG REQUIRED)
find_package(assimp REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory("${LFS_REPO_ROOT}/external" "${CMAKE_BINARY_DIR}/external" EXCLUDE_FROM_ALL)
add_subdirectory("${LFS_REPO_ROOT}/src/geometry" "${CMAKE_BINARY_DIR}/src/geometry")
add_subdirectory("${LFS_REPO_ROOT}/src/core" "${CMAKE_BINARY_DIR}/src/core")

add_library(lfs_python_utils STATIC stubs/python_runner_stub.cpp)
target_include_directories(lfs_python_utils PUBLIC "${LFS_REPO_ROOT}/src")
target_compile_features(lfs_python_utils PUBLIC cxx_std_23)
set_target_properties(lfs_python_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${LFS_REPO_ROOT}/src/io" "${CMAKE_BINARY_DIR}/src/io")
add_subdirectory("${LFS_REPO_ROOT}/src/training" "${CMAKE_BINARY_DIR}/src/training")

set(LFS_TRAIN_CLI_SOURCE "${LFS_REPO_ROOT}/tools/train_colmap_headless.cpp")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_colmap_headless.cpp")
    set(LFS_TRAIN_CLI_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tools/train_colmap_headless.cpp")
endif()

add_executable(train_colmap_headless "${LFS_TRAIN_CLI_SOURCE}")
target_include_directories(train_colmap_headless
    PRIVATE
        "${LFS_REPO_ROOT}/src"
        "${LFS_REPO_ROOT}/src/core/include"
        "${LFS_REPO_ROOT}/src/io/include"
        "${LFS_REPO_ROOT}/src/training/include"
        "${LFS_REPO_ROOT}/external"
        "${CMAKE_BINARY_DIR}/include"
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(train_colmap_headless
    PRIVATE
        lfs_training
        lfs_io
        lfs_core
        lfs_event_bridge
        CUDA::cudart
        spdlog::spdlog
        taywee::args
)

if(UNIX AND NOT APPLE)
    set_target_properties(train_colmap_headless PROPERTIES
        BUILD_RPATH "$ORIGIN;${CUDAToolkit_LIBRARY_DIR}"
    )
endif()

message(STATUS "Standalone COLMAP training CLI configured")
message(STATUS "  Repo root: ${LFS_REPO_ROOT}")
message(STATUS "  CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")

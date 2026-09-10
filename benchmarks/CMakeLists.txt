# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.24)

# Enhanced Message
if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(White       "${Esc}[37m")
  set(Cyan        "${Esc}[36m")
  set(BoldBlue    "${Esc}[1;34m")
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# Check if we are building using the TheRock install of ROCm.
set(USING_THE_ROCK OFF)
if (EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()

# Set device library path for TheRock install of ROCm.
if (USING_THE_ROCK)
    if (NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
        set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
    endif()
endif()

# Set HIP_PLATFORM to amd if not set. This is required before finding HIP.
if (NOT DEFINED ENV{HIP_PLATFORM})
    set(ENV{HIP_PLATFORM} "amd")
endif()

# Ensure CMake can find the roccv library and headers.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

project(roccv_bench VERSION 1.0 LANGUAGES CXX)

# Print benchmark CMake configuration information
message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColorReset}")

# Checked-in third-party sources; set by the root project, derived when
# benchmarks/ is configured standalone. See third_party/ORIGIN.md.
if(NOT DEFINED ROCCV_THIRD_PARTY_DIR)
    get_filename_component(ROCCV_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party" ABSOLUTE)
endif()

add_subdirectory(roccvbench)

set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "${ROCCV_THIRD_PARTY_DIR}/nlohmann_json/include")
set(BENCHMARK_LIBS ${BENCHMARK_LIBS} roccvbench-utils)

# Include roccv benchmark units
if (NOT TARGET roccv::roccv)
    find_package(roccv QUIET)
else()
    set(roccv_FOUND TRUE)
endif()

# Include rocRAND for rocCV benchmarks
find_package(rocrand QUIET)

# Dependency checks
set(BUILD_ROCCV_BENCHMARKS ON)
if (NOT roccv_FOUND)
    message("-- ${Red}${PROJECT_NAME} -- require roccv, but it was not found${ColorReset}")
    set(BUILD_ROCCV_BENCHMARKS OFF)
else()
    message("-- ${Green}${PROJECT_NAME} -- found roccv version: ${roccv_VERSION}${ColorReset}")
endif()

if (NOT rocrand_FOUND)
    message("-- ${Red}${PROJECT_NAME} -- require rocRAND, but it was not found${ColorReset}")
    set(BUILD_ROCCV_BENCHMARKS OFF)
else()
    message("-- ${Green}${PROJECT_NAME} -- found rocRAND version: ${rocrand_VERSION}${ColorReset}")
endif()

if (NOT BUILD_ROCCV_BENCHMARKS)
    message("-- ${Yellow}${PROJECT_NAME} -- Missing required dependencies for rocCV benchmarks. rocCV benchmarks will not be included in the benchmarks${ColorReset}")
else()
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv/*.cpp")
    set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv")
    set(BENCHMARK_LIBS ${BENCHMARK_LIBS} roc::rocrand roccv::roccv)
endif()

# Include OpenCV benchmark units
find_package(OpenCV QUIET)
if(${OpenCV_FOUND})
    message("-- ${Green}${PROJECT_NAME} -- Found OpenCV version: ${OpenCV_VERSION}${ColorReset}")
    set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/src/opencv/*.cpp")
    set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/src/opencv" ${OpenCV_INCLUDE_DIRS})
    set(BENCHMARK_LIBS ${BENCHMARK_LIBS} ${OpenCV_LIBRARIES})
else()
    message("-- ${Yellow}${PROJECT_NAME} -- Could not find OpenCV, it will not be included in the benchmarks ${ColorReset}")
endif()

file(GLOB BENCHMARK_SOURCES_GLOBBED
    CONFIGURE_DEPENDS    
    ${BENCHMARK_SOURCES}
)

add_executable(${PROJECT_NAME} ${BENCHMARK_SOURCES_GLOBBED} "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ${BENCHMARK_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${BENCHMARK_INCLUDES})
# In-tree libroccv first so we use the built library; ROCm lib remains for librocrand etc.
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "$<TARGET_FILE_DIR:roccv::roccv>;${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib"
)
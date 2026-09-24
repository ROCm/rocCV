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

# rocCV benchmarks. Built as part of the root project, or standalone against an
# installed rocCV.
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/RoccvBootstrap.cmake)

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

set(BENCHMARK_LIBS roccvbench-utils)
set(BENCHMARK_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib")

# Include roccv benchmark units
if(NOT TARGET roccv::roccv)
    find_package(roccv QUIET)
else()
    set(roccv_FOUND TRUE)
endif()

# Include rocRAND for rocCV benchmarks
find_package(rocrand QUIET)

# Dependency checks
set(BUILD_ROCCV_BENCHMARKS ON)
if(NOT roccv_FOUND)
    message("-- ${Red}${PROJECT_NAME} -- require roccv, but it was not found${ColorReset}")
    set(BUILD_ROCCV_BENCHMARKS OFF)
else()
    message("-- ${Green}${PROJECT_NAME} -- found roccv version: ${roccv_VERSION}${ColorReset}")
endif()

if(NOT rocrand_FOUND)
    message("-- ${Red}${PROJECT_NAME} -- require rocRAND, but it was not found${ColorReset}")
    set(BUILD_ROCCV_BENCHMARKS OFF)
else()
    message("-- ${Green}${PROJECT_NAME} -- found rocRAND version: ${rocrand_VERSION}${ColorReset}")
endif()

if(NOT BUILD_ROCCV_BENCHMARKS)
    message("-- ${Yellow}${PROJECT_NAME} -- Missing required dependencies for rocCV benchmarks. rocCV benchmarks will not be included in the benchmarks${ColorReset}")
else()
    list(APPEND BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv/*.cpp")
    list(APPEND BENCHMARK_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv")
    list(APPEND BENCHMARK_LIBS roc::rocrand roccv::roccv)
    # In-tree libroccv first so we use the built library; ROCm lib remains for librocrand etc.
    list(PREPEND BENCHMARK_RPATH "$<TARGET_FILE_DIR:roccv::roccv>")
endif()

# Include OpenCV benchmark units
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} -- Found OpenCV version: ${OpenCV_VERSION}${ColorReset}")
    list(APPEND BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/opencv/*.cpp")
    list(APPEND BENCHMARK_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/src/opencv" ${OpenCV_INCLUDE_DIRS})
    list(APPEND BENCHMARK_LIBS ${OpenCV_LIBRARIES})
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
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "${BENCHMARK_RPATH}"
)

# ##############################################################################
# Copyright (c)  - 2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.15)

# Enhanced Message
if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
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
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(roccv_bench VERSION 1.0 LANGUAGES CXX)

# Print benchmark CMake configuration information
message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColourReset}")

set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/vendor" "${CMAKE_CURRENT_SOURCE_DIR}/roccvbench/include")

# Include roccv benchmark units
if (NOT TARGET roccv::roccv)
    find_package(roccv REQUIRED)
endif()
message("-- ${Green}${PROJECT_NAME} -- Found rocCV version: ${roccv_VERSION}${ColourReset}")
set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv/*.cpp")
set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/src/roccv")
set(BENCHMARK_LIBS ${BENCHMARK_LIBS} roccv::roccv)

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
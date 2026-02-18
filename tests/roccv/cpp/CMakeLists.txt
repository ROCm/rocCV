# ##############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

if(NOT DEFINED ENHANCED_MESSAGE)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(White       "${Esc}[37m")
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

project(roccv-cpp-tests)

# Find required libraries
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# Check if we are building using the TheRock install of ROCm.
set(USING_THE_ROCK OFF)
if (EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()
message("-- ${White}${PROJECT_NAME} -- USING_THE_ROCK: ${USING_THE_ROCK}${ColourReset}")

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

# Find rocCV
if (NOT TARGET roccv::roccv)
    find_package(roccv REQUIRED)
endif()
message("-- ${Green}${PROJECT_NAME} -- Found rocCV version: ${roccv_VERSION}${ColourReset}")


# Compile test helper library
file(GLOB TEST_HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(roccv-test-helpers STATIC ${TEST_HELPER_SOURCES})
target_link_libraries(roccv-test-helpers PUBLIC roccv::roccv)
target_include_directories(roccv-test-helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Gather actual test sources
file(GLOB_RECURSE CPP_ROCCV_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp")

# Enable CTest
enable_testing()
include(CTest)

# Build each test as a separate executable
foreach(test_source ${CPP_ROCCV_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Get relative path from src/tests/ to determine output directory structure
    file(RELATIVE_PATH test_rel_path "${CMAKE_CURRENT_SOURCE_DIR}/src/tests" ${test_source})
    get_filename_component(test_subdir ${test_rel_path} DIRECTORY)
    
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE roccv-test-helpers)
    
    # Set output directory to match source structure: bin/tests/<subdir>/
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests/${test_subdir}"
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

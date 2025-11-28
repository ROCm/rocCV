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

# find HIP
find_package(HIP REQUIRED)
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${HIP_INCLUDE_DIR})
# find rocCV
find_library(roccv_LIBRARY roccv PATHS ${ROCM_PATH}/lib)
find_path(roccv_INCLUDE_DIR NAMES roccv_operators.hpp PATHS ${ROCM_PATH}/include/roccv)

# Add current source directory as an include
set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# If building from source, ensure we use the locally built version of roccv. Otherwise, look for an installed version to link against.
if(BUILD_FROM_SOURCE)
        message("-- ${Green}${PROJECT_NAME} -- Building CPP tests using roccv source build${ColourReset}")
        set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${CMAKE_SOURCE_DIR}/include)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} roccv)
else()
    if(roccv_LIBRARY AND roccv_INCLUDE_DIR)
        message("-- ${Green}${PROJECT_NAME} -- Found rocCV \n\troccv_LIBRARY:${roccv_LIBRARY} \n\troccv_INCLUDE_DIR:${roccv_INCLUDE_DIR}${ColourReset}")
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${roccv_LIBRARY})
        set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${roccv_INCLUDE_DIR})
        set(roccv_FOUND TRUE)
    else()
        message(FATAL_ERROR "${Red}Find rocCV -- rocCV NOT FOUND${ColourReset}")
    endif()
endif()

# Gather test helper sources
file(GLOB TEST_HELPER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create a static library for test helpers (so we don't recompile for every test)
add_library(roccv-test-helpers STATIC ${TEST_HELPER_SOURCES})
target_link_libraries(roccv-test-helpers PUBLIC ${LINK_LIBRARY_LIST})
target_include_directories(roccv-test-helpers PUBLIC ${INCLUDE_DIR_LIST})

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

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

project(roccv-test-helper)

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

# find HIP
find_package(HIP REQUIRED)
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${HIP_INCLUDE_DIR})
# find rocCV
find_library(roccv_LIBRARY roccv PATHS ${ROCM_PATH}/lib)
find_path(roccv_INCLUDE_DIR NAMES roccv_operators.hpp PATHS ${ROCM_PATH}/include/roccv)

# Add current source directory as an include
set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${CMAKE_CURRENT_SOURCE_DIR})

# If building from source, ensure we use the locally built version of roccv. Otherwise, look for an installed version to link against.
if(BUILD_FROM_SOURCE)
        message("-- ${Green}${PROJECT_NAME} -- Building CPP tests using roccv source build${ColourReset}")
        set(INCLUDE_DIR_LIST ${INCLUDE_DIR_LIST} ${ROCCV_SOURCE_DIR}/include)
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

file(GLOB TEST_HELPER_SOURCES "*.cpp")

# Create roccv-test-helpers library
add_library(${PROJECT_NAME} STATIC ${TEST_HELPER_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIR_LIST})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LINK_LIBRARY_LIST})
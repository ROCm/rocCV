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

# Enable enhanced message
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

set(VERSION "1.0")
project(roccv-samples VERSION ${VERSION} LANGUAGES CXX)

message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColourReset}")

# Function for defining a sample executable
function(define_sample_executable NAME)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE roccv::roccv ${OpenCV_LIBS})
    target_include_directories(${NAME} PRIVATE ${Open_CV_INCLUDE_DIRS})
endfunction()

if (NOT TARGET roccv::roccv)
    find_package(roccv QUIET)
else()
    set(roccv_FOUND TRUE)
endif()

find_package(OpenCV QUIET)

set(BUILD_SAMPLES ON)
if (NOT roccv_FOUND)
    message("-- ${Red}roccv-samples -- require roccv, but it was not found${ColourReset}")
    set(BUILD_SAMPLES OFF)
else()
    message("-- ${Green}roccv-samples -- found roccv version: ${roccv_VERSION}${ColourReset}")
endif()
if (NOT OpenCV_FOUND)
    message("-- ${Red}roccv-samples -- require OpenCV, but it was not found${ColourReset}")
    set(BUILD_SAMPLES OFF)
else()
    message("-- ${Green}roccv-samples -- found OpenCV version: ${OpenCV_VERSION}${ColourReset}")
endif()

if (BUILD_SAMPLES)
    define_sample_executable(bilateral_filter)
    define_sample_executable(custom_crop)
    define_sample_executable(center_crop)
    define_sample_executable(normalize)
    define_sample_executable(bnd_box)
    define_sample_executable(copy_make_border)
    define_sample_executable(warp_perspective)
    define_sample_executable(composite)
    define_sample_executable(gamma_contrast)
    add_subdirectory(cropandresize/cpp)
else()
    message("-- ${Yellow}roccv-samples -- missing required dependencies, will not be built${ColourReset}")
endif()
# add_subdirectory(cropandresize/cpp)
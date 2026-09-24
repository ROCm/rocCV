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

# Setup shared by the rocCV CMake projects that are configured from the source
# tree (the root project, python/ and benchmarks/). Include it before project()
# so the compiler and build type defaults take effect.
#
# NOTE: samples/ and tests/roccv/cpp/ are also configured standalone from the
#       install tree, where this file is not available, so they carry their own
#       copy of this logic. Keep them in sync with this file.

include_guard(GLOBAL)

# Colored status messages
if(NOT DEFINED ENHANCED_MESSAGE OR ENHANCED_MESSAGE)
    string(ASCII 27 Esc)
    set(ColorReset "${Esc}[m")
    set(Red        "${Esc}[31m")
    set(Green      "${Esc}[32m")
    set(Yellow     "${Esc}[33m")
    set(Blue       "${Esc}[34m")
    set(BoldBlue   "${Esc}[1;34m")
    set(Magenta    "${Esc}[35m")
    set(Cyan       "${Esc}[36m")
    set(White      "${Esc}[37m")
endif()

# ROCm installation path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Default to a Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type [options: Debug/Release]" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

# C++20, with AMD Clang as the default compiler
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TheRock installs of ROCm keep the device libraries under lib/llvm.
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()
if(USING_THE_ROCK AND NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
endif()

# HIP_PLATFORM must be set before finding HIP.
if(NOT DEFINED ENV{HIP_PLATFORM})
    set(ENV{HIP_PLATFORM} "amd")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

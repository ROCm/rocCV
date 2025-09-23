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
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

project(roccv-test-helper)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# find HIP
find_package(HIP REQUIRED)
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
# find opencv
find_package(OpenCV REQUIRED)
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCV_LIBRARIES})
# find rocCV
find_library(roccv_LIBRARY roccv PATHS ${ROCM_PATH}/lib)
find_path(roccv_INCLUDE_DIR NAMES roccv_operators.hpp PATHS ${ROCM_PATH}/include/roccv)
if(roccv_LIBRARY AND roccv_INCLUDE_DIR)
    message("-- ${Green}${PROJECT_NAME} -- Found rocCV \n\troccv_LIBRARY:${roccv_LIBRARY} \n\troccv_INCLUDE_DIR:${roccv_INCLUDE_DIR}${ColourReset}")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${roccv_LIBRARY})
else()
    if(BUILD_FROM_SOURCE)
        message("-- ${Yellow}${PROJECT_NAME} -- Install rocCV before testing${ColourReset}")
        include_directories(${CMAKE_SOURCE_DIR}/include)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} roccv)
    else()
        message(FATAL_ERROR "{Red}Find rocCV -- rocCV NOT FOUND${ColourReset}")
    endif()
endif()

file(GLOB TEST_HELPER_SOURCES "*.cpp")
add_library(${PROJECT_NAME} STATIC ${TEST_HELPER_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_INCLUDE_DIR} ${roccv_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LINK_LIBRARY_LIST})
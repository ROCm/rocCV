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

set(VERSION "1.0")
project(roccv-samples VERSION ${VERSION} LANGUAGES CXX)

message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColourReset}")

# Copy sample images to relative data directory
set(DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
# TODO: This will not work in standalone build, must fix this.
file(GLOB SAMPLE_DATA_FILES "${ROCCV_SOURCE_DIR}/data/samples/*")

# Function for defining a sample executable
function(define_sample_executable NAME)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE roccv::roccv ${OpenCV_LIBS})
    target_include_directories(${NAME} PRIVATE ${Open_CV_INCLUDE_DIRS})
endfunction()

file(MAKE_DIRECTORY ${DESTINATION_DIR})
foreach(FILE ${SAMPLE_DATA_FILES})
    file(COPY ${FILE} DESTINATION ${DESTINATION_DIR})
endforeach()

find_package(roccv QUIET)
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
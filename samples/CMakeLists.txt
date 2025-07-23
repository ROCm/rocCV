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

# Copy sample images to relative data directory
set(DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(GLOB SAMPLE_DATA_FILES "${CMAKE_SOURCE_DIR}/data/samples/*")

include_directories(${CMAKE_SOURCE_DIR}/include ${ROCM_PATH}/include)

file(MAKE_DIRECTORY ${DESTINATION_DIR})
foreach(FILE ${SAMPLE_DATA_FILES})
    file(COPY ${FILE} DESTINATION ${DESTINATION_DIR})
endforeach()

add_executable(normalize normalize.cpp)
target_link_libraries(normalize PRIVATE roccv ${OpenCV_LIBS})
target_include_directories(normalize PRIVATE ${Open_CV_INCLUDE_DIRS})

add_executable(bnd_box bnd_box.cpp)
target_link_libraries(bnd_box PRIVATE roccv ${OpenCV_LIBS})
target_include_directories(bnd_box PRIVATE ${Open_CV_INCLUDE_DIRS})

add_executable(copy_make_border copy_make_border.cpp)
target_link_libraries(copy_make_border PRIVATE roccv ${OpenCV_LIBS})
target_include_directories(copy_make_border PRIVATE ${Open_CV_INCLUDE_DIRS})

add_executable(warp_perspective warp_perspective.cpp)
target_link_libraries(warp_perspective PRIVATE roccv ${OpenCV_LIBS})
target_include_directories(warp_perspective PRIVATE ${Open_CV_INCLUDE_DIRS})

add_executable(composite composite.cpp)
target_link_libraries(composite PRIVATE roccv ${OpenCV_LIBS})
target_include_directories(composite PRIVATE ${Open_CV_INCLUDE_DIRS})

add_subdirectory(cropandresize/cpp)
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

project(roccv_bench)

# Benchmark suite helper libraries
add_subdirectory("roccvbench")

set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "vendor")
set(BENCHMARK_LIBS ${BENCHMARK_LIBS} roccv_bench_utils)

# Include roccv benchmark units
set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} "src/roccv/*.cpp")

# Include OpenCV benchmark units
set(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} "src/opencv/*.cpp")
set(BENCHMARK_INCLUDES ${BENCHMARK_INCLUDES} "src/opencv" ${OpenCV_INCLUDE_DIRS})
set(BENCHMARK_LIBS ${BENCHMARK_LIBS} ${OpenCV_LIBRARIES})

file(GLOB BENCHMARK_SOURCES_GLOBBED
    CONFIGURE_DEPENDS    
    ${BENCHMARK_SOURCES}
)

add_executable(${PROJECT_NAME} ${BENCHMARK_SOURCES_GLOBBED} "src/main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ${BENCHMARK_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${BENCHMARK_INCLUDES})
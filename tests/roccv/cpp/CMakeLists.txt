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

file(GLOB CPP_ROCCV_TEST_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.cpp") 
create_test_sourcelist(CppRocCVTests 
    system_test_main.cpp 
    ${CPP_ROCCV_TEST_SOURCES})

add_executable(RocCVTests ${CppRocCVTests})
target_link_libraries(RocCVTests PRIVATE TestHelpers)

set(TestsToRun ${CppRocCVTests})
remove (TestsToRun system_test_main.cpp)

foreach (test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND RocCVTests ${TName} "${CMAKE_SOURCE_DIR}/data")
endforeach()
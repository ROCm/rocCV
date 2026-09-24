################################################################################
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
################################################################################
cmake_minimum_required(VERSION 3.24)

# rocCV Python binding tests. Configured standalone against an installed rocCV:
#   cmake ${ROCM_PATH}/share/roccv/test/pybind
project(roccv-pybind-test LANGUAGES NONE)

# make test with CTest
include(CTest)

# ROCm installation path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- ${PROJECT_NAME} INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

if(PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET COMPONENTS Interpreter)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter)
endif()

set(ROCCV_PYBIND_SCRIPTS OFF)
if(EXISTS "${ROCM_PATH}/lib/rocpycv.pyi")
    set(ROCCV_PYBIND_SCRIPTS ON)
endif()

if(NOT Python3_Interpreter_FOUND OR NOT ROCCV_PYBIND_SCRIPTS)
    if(NOT Python3_Interpreter_FOUND)
        message("-- NOTE: rocCV pybind requires Python3 - NOT FOUND")
    endif()
    if(NOT ROCCV_PYBIND_SCRIPTS)
        message("-- NOTE: rocCV pybind requires rocpycv project built and installed - NOT FOUND")
    endif()
    return()
endif()

message("-- ${PROJECT_NAME}: rocCV Pybind found at ${ROCM_PATH}/lib/rocpycv")

# Function test
add_test(NAME roccv_pybind_test_functions
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/rocpycv_test.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set(PYBIND_TESTS roccv_pybind_test_functions)

# One test per operator, e.g. python/test_op_resize.py -> roccv_pybind_test_op_resize
file(GLOB OPERATOR_TEST_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_*.py")
list(SORT OPERATOR_TEST_SCRIPTS)
foreach(test_script ${OPERATOR_TEST_SCRIPTS})
    get_filename_component(test_name ${test_script} NAME_WE)
    add_test(NAME roccv_pybind_${test_name}
        COMMAND ${Python3_EXECUTABLE} -m pytest ${test_script} --rootdir ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    list(APPEND PYBIND_TESTS roccv_pybind_${test_name})
endforeach()

set_tests_properties(${PYBIND_TESTS} PROPERTIES
    ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${ROCM_PATH}/lib"
)

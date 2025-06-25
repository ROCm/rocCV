################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.15)

project(roccv-pybind-test)

# make test with CTest
enable_testing()
include(CTest)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- ${PROJECT_NAME} INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

if(PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET)
endif()

set(ROCCV_PYBIND_SCRIPTS OFF)
if(EXISTS "${ROCM_PATH}/lib/rocpycv/rocpycv.pyi")
  set(ROCCV_PYBIND_SCRIPTS ON)
endif()

set(PYTEST_FLAGS  
    "--data_dir" "${ROCM_PATH}/share/roccv/test/data/tests/ops"
    "--rootdir" "${CMAKE_CURRENT_BINARY_DIR}"
)

# rocCV PyBind Tests
if(Python3_FOUND AND ROCCV_PYBIND_SCRIPTS)
  message("-- ${PROJECT_NAME}: rocCV Pybind found at ${ROCM_PATH}/lib/rocpycv")
  # 1 - function test
  add_test(NAME roccv_pybind_test_functions
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/rocpycv_test.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_functions PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 2 - Test:OP -- Thresholding
  add_test(NAME roccv_pybind_test_op_thresholding
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_thresholding.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_thresholding PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 3 - Test:OP -- Bilateral Filter
  add_test(NAME roccv_pybind_test_op_bilateral_filter
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_bilateral_filter.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_bilateral_filter PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 4 - Test:OP -- Bounding Box
  add_test(NAME roccv_pybind_test_op_bnd_box
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_bnd_box.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_bnd_box PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 5 - Test:OP -- Histogram
  add_test(NAME roccv_pybind_test_op_histogram
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_histogram.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_histogram PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 6 - Test:OP -- Warp Perspective
  add_test(NAME roccv_pybind_test_op_warp_perspective
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_warp_perspective.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_warp_perspective PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 7 - Test:OP -- Non Max Suppression
  add_test(NAME roccv_pybind_test_op_non_max_suppression
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_non_max_suppression.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_non_max_suppression PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 8 - Test:OP -- Rotate
  add_test(NAME roccv_pybind_test_op_rotate
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_rotate.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_rotate PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 9 - Test:OP -- Normalize
  add_test(NAME roccv_pybind_test_op_normalize
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_normalize.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_normalize PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 10 - Test:OP -- Color Convert
  add_test(NAME roccv_pybind_test_op_cvt_color
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_cvt_color.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_cvt_color PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 11 - Test:OP -- Composite
  add_test(NAME roccv_pybind_test_op_composite
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_composite.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_composite PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 12 - Test:OP -- Custom Crop
  add_test(NAME roccv_pybind_test_op_custom_crop
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_custom_crop.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_custom_crop PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 13 - Test:OP -- Flip
  add_test(NAME roccv_pybind_test_op_flip
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_flip.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_flip PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 14 - Test:OP -- Gamma Contrast
  add_test(NAME roccv_pybind_test_op_gamma_contrast
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_gamma_contrast.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_gamma_contrast PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 15 - Test:OP -- Remap
  add_test(NAME roccv_pybind_test_op_remap
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_remap.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_remap PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 16 - Test:OP -- Resize
  add_test(NAME roccv_pybind_test_op_resize
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_resize.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_resize PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 17 - Test:OP -- Warp Affine
  add_test(NAME roccv_pybind_test_op_warp_affine
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_warp_affine.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_warp_affine PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 18 - Test:OP -- CopyMakeBorder
  add_test(NAME roccv_pybind_test_op_copy_make_border
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_copy_make_border.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_copy_make_border PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

  # 19 - Test:OP -- Center Crop
  add_test(NAME roccv_pybind_test_op_center_crop
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/test_op_center_crop.py ${PYTEST_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST roccv_pybind_test_op_center_crop PROPERTY ENVIRONMENT "PYTHONPATH=${ROCM_PATH}/lib:$PYTHONPATH")

else()
  if(NOT Python3_FOUND)
    message("-- NOTE: rocCV pybind requires Python3 - NOT FOUND")
  endif()
  if(NOT ROCCV_PYBIND_SCRIPTS)
    message("-- NOTE: rocCV pybind requires rocpycv project built and installed - NOT FOUND")
  endif()
endif(Python3_FOUND AND ROCCV_PYBIND_SCRIPTS)

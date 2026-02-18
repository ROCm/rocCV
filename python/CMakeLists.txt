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

# rocCV pybind wheel install
cmake_minimum_required(VERSION 3.28)

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
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# pyVersion
# NOTE: rocCV version updates requires updates in
#       * rocCV source CMakeLists.txt
#       * python CMakeLists.txt
set(VERSION "0.4.0")
project(rocpycv VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

# --- Find dependencies ---
# Auto-detect Python3 executable if not specified
if(NOT Python3_EXECUTABLE AND NOT Python3_ROOT_DIR AND NOT PYTHON_VERSION_SUGGESTED)
    find_program(PYTHON3_EXECUTABLE_PATH python3 QUIET)
    if(PYTHON3_EXECUTABLE_PATH)
        set(Python3_EXECUTABLE ${PYTHON3_EXECUTABLE_PATH})
        get_filename_component(Python3_ROOT_DIR "${PYTHON3_EXECUTABLE_PATH}" DIRECTORY)
        get_filename_component(Python3_ROOT_DIR "${Python3_ROOT_DIR}" DIRECTORY)
    endif()
endif()

if(PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET COMPONENTS Interpreter Development)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
endif()

# --- Fetch content ---
include(FetchContent)

# --- pybind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.2
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pybind11)

# --- dlpack ---
FetchContent_Declare(
    dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack.git
    GIT_TAG v1.3
    GIT_SHALLOW TRUE
    CMAKE_ARGS -DBUILD_MOCK=OFF
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(dlpack)

# --- Dependency checks ---
set(BUILD_ROCCV_PYPACKAGE true)

# Python3
if(NOT Python3_Interpreter_FOUND)
    set(BUILD_ROCCV_PYPACKAGE OFF)
    message("-- ${Red}NOTE: ${PROJECT_NAME} requires the python${PYTHON_VERSION_SUGGESTED} interpreter, Not Found${ColourReset}")
else()
    message("-- ${Green}${PROJECT_NAME} -- Found python${PYTHON_VERSION_SUGGESTED} interpreter: ${Python3_EXECUTABLE}${ColourReset}")
endif()

# Python3 Development
if(NOT Python3_Development_FOUND)
    set(BUILD_ROCCV_PYPACKAGE OFF)
    message("-- ${Red}NOTE: rocpycv requires the python${PYTHON_VERSION_SUGGESTED} development headers, Not Found${ColourReset}")
else()
    message("-- ${Green}${PROJECT_NAME} -- Found python${PYTHON_VERSION_SUGGESTED} development headers: ${Python3_INCLUDE_DIRS}${ColourReset}")
endif()

# Check that all dependencies have been met
if(NOT BUILD_ROCCV_PYPACKAGE)
	message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} dependencies not satisfied${ColourReset}")
endif()

file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
pybind11_add_module(${PROJECT_NAME} MODULE ${sources} WITH_SOABI)
target_include_directories(${PROJECT_NAME} PRIVATE ${pybind11_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include" "${ROCCV_SOURCE_DIR}/include")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        roccv
        dlpack::dlpack
        pybind11::headers
)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN;$ORIGIN/../rocm_sysdeps/lib;${ROCM_PATH}/lib/llvm/lib" BUILD_WITH_INSTALL_RPATH TRUE)

# install rocpycv lib -- {ROCM_PATH)/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME} COMPONENT runtime)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/rocpycv DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
message("-- ${Green}rocCV Python Binding - rocpycv module added ${ColourReset}")

# PyBind Test - Installed
install(DIRECTORY ${ROCCV_SOURCE_DIR}/tests/pybind
        DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test
        COMPONENT test
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN ".pytest_cache" EXCLUDE)

install(DIRECTORY ${ROCCV_SOURCE_DIR}/tests/roccv/python
        DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test/pybind
        COMPONENT test
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN ".pytest_cache" EXCLUDE)

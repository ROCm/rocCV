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

# pyVersion
# NOTE: rocCV version updates requires updates in 
#       * rocCV source CMakeLists.txt
#       * python CMakeLists.txt
#       * pyproject.toml
set(VERSION "0.2.0")
project(rocpycv VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

if (PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET COMPONENTS Interpreter Development)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
endif()

# Locate cmake directory for pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH $ENV{PROJECT_ROOT}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

find_package(HIP QUIET)
find_package(pybind11 CONFIG REQUIRED)
find_package(dlpack REQUIRED)

set(BUILD_ROCCV_PYPACKAGE true)

# Dependency checks
# HIP
if(NOT HIP_FOUND)
    set(BUILD_ROCCV_PYPACKAGE false)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires HIP, Not Found${ColourReset}")
endif()
# DLPack
if (NOT dlpack_FOUND)
	set(BUILD_ROCCV_PYPACKAGE false)
	message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires DLPack, Not Found ${ColourReset}")
endif()
# Pybind11
if (NOT pybind11_FOUND)
	set(BUILD_ROCCV_PYPACKAGE false)
	message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires pybind11, Not Found ${ColourReset}")
endif()

if (NOT BUILD_ROCCV_PYPACKAGE)
    message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} Python Package dependencies not satisfied${ColourReset}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${ROCM_PATH}/include)

file(GLOB_RECURSE sources "src/*.cpp")
pybind11_add_module(${PROJECT_NAME} MODULE ${sources} WITH_SOABI)
target_include_directories(${PROJECT_NAME} PRIVATE ${pybind11_INCLUDE_DIRS} "include")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(${PROJECT_NAME} PRIVATE roccv pybind11::headers dlpack::dlpack)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")

# install rocpycv lib -- {ROCM_PATH)/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME} COMPONENT runtime)
install(DIRECTORY src/rocpycv DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
message("-- ${Green}rocCV Python Binding - rocpycv module added ${ColourReset}")

# PyBind Test - Installed
install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test COMPONENT test)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/pybind DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test COMPONENT test)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/roccv/python DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test/pybind COMPONENT test)

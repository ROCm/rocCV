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
cmake_minimum_required(VERSION 3.24)

# Enhanced Message
if(NOT DEFINED ENHANCED_MESSAGE)
    string(ASCII 27 Esc)
    set(ColorReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(White       "${Esc}[37m")
    set(Cyan        "${Esc}[36m")
    set(BoldBlue    "${Esc}[1;34m")
endif()

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
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# Check if we are building using the TheRock install of ROCm.
set(USING_THE_ROCK OFF)
if (EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()

# Set device library path for TheRock install of ROCm.
if (USING_THE_ROCK)
    if (NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
        set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
    endif()
endif()

# Set HIP_PLATFORM to amd if not set. This is required before finding HIP.
if (NOT DEFINED ENV{HIP_PLATFORM})
    set(ENV{HIP_PLATFORM} "amd")
endif()

# Ensure CMake can find the roccv library and headers.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# pyVersion
# NOTE: rocCV version updates requires updates in
#       * rocCV source CMakeLists.txt
#       * python CMakeLists.txt
set(VERSION "0.4.0")
project(rocpycv VERSION ${VERSION} LANGUAGES CXX)

message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${PROJECT_VERSION}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Compiler -- ${CMAKE_CXX_COMPILER}${ColorReset}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

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
set(FETCHCONTENT_QUIET OFF)
message("-- ${BoldBlue}${PROJECT_NAME} -- Fetching pybind11 v3.0.2${ColorReset}")
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.2.tar.gz
    URL_HASH SHA256=2f20a0af0b921815e0e169ea7fec63909869323581b89d7de1553468553f6a2d
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pybind11)

# --- dlpack ---
message("-- ${BoldBlue}${PROJECT_NAME} -- Fetching dlpack v1.3${ColorReset}")
FetchContent_Declare(
    dlpack
    URL https://github.com/dmlc/dlpack/archive/refs/tags/v1.3.tar.gz
    URL_HASH SHA256=f3d567f885f6c142183afc91a58873b31d0e0b36faa2e45c232b98c74596404f
    CMAKE_ARGS -DBUILD_MOCK=OFF
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(dlpack)
set(FETCHCONTENT_QUIET ON)

if (NOT TARGET roccv::roccv)
    find_package(roccv QUIET)
else()
    set(roccv_FOUND TRUE)
endif()

# --- Dependency checks ---
set(BUILD_ROCCV_PYPACKAGE true)

# rocCV
if (NOT roccv_FOUND)
    message("-- ${Red}${PROJECT_NAME} -- require roccv, but it was not found${ColorReset}")
    set(BUILD_ROCCV_PYPACKAGE OFF)
else()
    message("-- ${Green}${PROJECT_NAME} -- found roccv version: ${roccv_VERSION}${ColorReset}")
endif()

# Python3
if(NOT Python3_Interpreter_FOUND)
    set(BUILD_ROCCV_PYPACKAGE OFF)
    message("-- ${Red}NOTE: ${PROJECT_NAME} requires the python${PYTHON_VERSION_SUGGESTED} interpreter, Not Found${ColorReset}")
else()
    message("-- ${Green}${PROJECT_NAME} -- Found python${PYTHON_VERSION_SUGGESTED} interpreter: ${Python3_EXECUTABLE}${ColorReset}")
endif()

# Python3 Development
if(NOT Python3_Development_FOUND)
    set(BUILD_ROCCV_PYPACKAGE OFF)
    message("-- ${Red}NOTE: rocpycv requires the python${PYTHON_VERSION_SUGGESTED} development headers, Not Found${ColorReset}")
else()
    message("-- ${Green}${PROJECT_NAME} -- Found python${PYTHON_VERSION_SUGGESTED} development headers: ${Python3_INCLUDE_DIRS}${ColorReset}")
endif()

# Check that all dependencies have been met
if(NOT BUILD_ROCCV_PYPACKAGE)
	message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} dependencies not satisfied${ColorReset}")
endif()

file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
pybind11_add_module(${PROJECT_NAME} MODULE ${sources} WITH_SOABI)
target_include_directories(${PROJECT_NAME} PRIVATE ${pybind11_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        roccv::roccv
        dlpack::dlpack
        pybind11::headers
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
    BUILD_RPATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR};${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib"
    INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN;\$ORIGIN/../rocm_sysdeps/lib;\$ORIGIN/../llvm/lib"
)

# install rocpycv lib -- {ROCM_PATH)/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rocpycv.pyi DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
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

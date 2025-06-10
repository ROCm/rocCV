# ##############################################################################
# Copyright (c)  - 2024 Advanced Micro Devices, Inc.
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

# NOTE: rocCV version updates requires updates in 
#       * rocCV source CMakeLists.txt
#       * python CMakeLists.txt
#       * pyproject.toml
set(VERSION "0.2.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Project Version and Language
project(roccv VERSION ${VERSION} LANGUAGES CXX C)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "rocCV default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# rocCV Default Options
option(ENHANCED_MESSAGE         "rocCV Enhanced Message Option"                 ON)
option(BUILD_PYPACKAGE          "Build with rocCV Python Package"               ON)
option(BUILD_WITH_AMD_ADVANCE   "Build rocCV for advanced AMD GPU Architecture" OFF)
option(FULL_BUILD               "Build with all tests and samples"              OFF)
option(SAMPLES                  "Build with samples"                            ON)
option(TESTS                    "Build with tests"                              ON)
option(BENCHMARKS               "Build with benchmarks"                         OFF)
set(PYTHON_VERSION_SUGGESTED    ""          CACHE STRING "Python version to build rocCV")
set(DEFAULT_BUILD_TYPE          "Release"   CACHE STRING "Default build type for rocCV")

if(ENHANCED_MESSAGE)
	string(ASCII 27 Esc)
	set(ColourReset "${Esc}[m")
	set(Red "${Esc}[31m")
	set(Green "${Esc}[32m")
	set(Yellow "${Esc}[33m")
	set(Blue "${Esc}[34m")
	set(BoldBlue "${Esc}[1;34m")
	set(Magenta "${Esc}[35m")
	set(Cyan "${Esc}[36m")
	set(White "${Esc}[37m")
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

# rocCV Build Type
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE} CACHE STRING "rocCV Default Build Type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
	# -O0 		-- Don't Optimize output file 
	# -g  		-- generate debugging information --- dwarf-4 for making valgrind work
    # -Wall     -- Print all warnings
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4 -Wall")
else()
	# -O3       -- Optimize output file 
	# -DNDEBUG  -- turn off asserts 
	# -fPIC     -- Generate position-independent code if possible
    # -Wall     -- Print all warnings
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC -Wall")
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

# Set all build flags if FULL_BUILD is set.
if(FULL_BUILD)
    set(SAMPLES ON)
    set(TESTS ON)
endif()

message("-- ${Cyan}${PROJECT_NAME} Developer Options${ColourReset}")
message("-- ${Cyan}     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} [Select rocCV build type [options:Debug/Release](default:Release)]${ColourReset}")
message("-- ${Cyan}     -D FULL_BUILD=${FULL_BUILD} [Compile with samples, and tests [options:ON/OFF](default:OFF)]${ColourReset}")
message("-- ${Cyan}     -D SAMPLES=${SAMPLES} [Compile with samples [options:ON/OFF](default:ON)]${ColourReset}")
message("-- ${Cyan}     -D TESTS=${TESTS} [Compile with tests [options:ON/OFF](default:ON)]${ColourReset}")
message("-- ${Cyan}     -D BENCHMARKS=${BENCHMARKS} [Compile with benchmarks [options:ON/OFF](default:OFF)]${ColourReset}")
message("-- ${Cyan}     -D BUILD_PYPACKAGE=${BUILD_PYPACKAGE} [Turn ON/OFF Build for rocCV Python Package(default:ON)]${ColourReset}")
message("-- ${Cyan}     -D PYTHON_VERSION_SUGGESTED=${PYTHON_VERSION_SUGGESTED} [User provided python version to use for rocCV Python Bindings(default:System Version)]${ColourReset}")

# Set supported GPU Targets
set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=native")

# Set AMD GPU_TARGETS
if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
    message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
    if(DEFINED ENV{AMDGPU_TARGETS})
        set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
    else()
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
endif()

if(DEFINED ENV{GPU_ARCHS})
    set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
elseif(GPU_TARGETS)
    message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
else()
    set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")

# Do not warn on unused function results.
add_compile_options("-Wno-unused-result")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

# Find dependencies
find_package(HIP QUIET)
find_package(StdFilesystem QUIET)
find_package(rocm_smi QUIET)
if(PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET COMPONENTS Interpreter Development)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
endif()
find_package(OpenCV QUIET)

# Dependency checks
set(BUILD_ROCCV true)
# HIP
if(NOT HIP_FOUND)
    set(BUILD_ROCCV false)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires HIP, Not Found${ColourReset}")
endif()
# Filesystem
if(NOT FILESYSTEM_FOUND)
    set(BUILD_ROCCV false)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires FileSystem, Not Found ${ColourReset}")	
endif()
# rocm_smi
if (NOT rocm_smi_FOUND)
	set(BUILD_ROCCV false)
	message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires rocm_smi, Not Found ${ColourReset}")
endif()

# Python3	
if(NOT Python3_FOUND)	
    set(BUILD_PYPACKAGE OFF)	
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} rocPyCV requires Python3, Not Found${ColourReset}")
endif()
# OpenCV
if(NOT OpenCV_FOUND)
    set(FULL_BUILD OFF)
    set(SAMPLES OFF)
    set(TESTS OFF)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} samples, and tests requires OpenCV, Not Found ${ColourReset}")
endif()

# Check that all dependencies have been met
if (NOT BUILD_ROCCV)
	message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} dependencies not satisfied${ColourReset}")
endif()

add_subdirectory(src)

if(SAMPLES)
    add_subdirectory(samples)
endif(SAMPLES)

if(BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BENCHMARKS)

# Enable output to stdout by default when tests fail
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
enable_testing()

if(TESTS)
	add_subdirectory(tests)
	add_custom_target(test_correctness
		COMMAND ${CMAKE_CTEST_COMMAND} -R "^test_" --output-on-failure
        COMMENT "Running correctness tests for rocCV."
	)
endif(TESTS)

if(BUILD_PYPACKAGE)
    add_subdirectory("${CMAKE_SOURCE_DIR}/python")
endif(BUILD_PYPACKAGE)

# set package information
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "MIVisionX Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCm/rocCV")

if(ENABLE_ASAN_PACKAGING)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocCV address sanitizer libraries")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocCV address sanitizer libraries")
else()
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocCV is a comprehensive Computer Vision and ML Inference Toolkit")
    set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocCV is a comprehensive Computer Vision and ML Inference Toolkit")
endif()

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

# set license information
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# set license information - {ROCM_PATH}/share/doc/roccv
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# Set the dependent packages
# * Runtime
set(ROCCV_RUNTIME_PACKAGE_LIST  "rocm-hip-runtime")
# * Dev
set(ROCCV_DEBIAN_DEV_PACKAGE_LIST  "rocm-hip-runtime-dev, python3-dev")
set(ROCCV_RPM_DEV_PACKAGE_LIST  "rocm-hip-runtime-devel, python3-devel")
# * Test
set(ROCCV_DEBIAN_TEST_PACKAGE_LIST  "python3-pytest, python3-numpy, python3-opencv")
set(ROCCV_RPM_TEST_PACKAGE_LIST  "python3-pytest, python3-numpy")

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(
    COMMAND rpm --eval %{?dist}
    RESULT_VARIABLE PROC_RESULT
    OUTPUT_VARIABLE EVAL_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

# set package 
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV "no")
# Debian package - COMPONENT
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${ROCCV_RUNTIME_PACKAGE_LIST}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}, ${ROCCV_DEBIAN_DEV_PACKAGE_LIST}")
# Debian package - specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan, ${ROCCV_RUNTIME_PACKAGE_LIST}" )
# Debian package - Test
set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}, ${ROCCV_DEBIAN_TEST_PACKAGE_LIST}" )
# RPM package - COMPONENT
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${ROCCV_RUNTIME_PACKAGE_LIST}")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_RUNTIME_PACKAGE_NAME}, ${ROCCV_RPM_DEV_PACKAGE_LIST}")
# RPM package - specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${ROCCV_RUNTIME_PACKAGE_LIST}" )
# RPM package specific variable for ASAN
set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}, ${ROCCV_RPM_TEST_PACKAGE_LIST}" )

if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
endif()

if(ENABLE_ASAN_PACKAGING)
    # ASAN Package requires asan component with only libraries and license file
    set(CPACK_COMPONENTS_ALL asan)
else()
    set(CPACK_COMPONENTS_ALL runtime dev test)
endif()

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE "local")

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

set(CPACK_GENERATOR "TGZ;ZIP")
if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
endif()
if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
    message("-- ${White}AMD ROCm rocCV rpm RunTime Package -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}AMD ROCm rocCV rpm Dev Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}AMD ROCm rocCV rpm Test Package -- ${CPACK_RPM_TEST_PACKAGE_REQUIRES}${ColourReset}")
endif()
if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}AMD ROCm rocCV deb RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}AMD ROCm rocCV deb Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}AMD ROCm rocCV deb Test Package -- ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS}${ColourReset}")
endif()

include(CPack)

cpack_add_component(runtime
                DISPLAY_NAME "rocCV Runtime Package"
                DESCRIPTION "AMD rocCV is an efficient GPU-accelerated library for pre and post-processing of images using AMD’s HIP platform. \
rocCV runtime package provides rocCV libraries and license.txt")

cpack_add_component(dev
                DISPLAY_NAME "rocCV Develop Package"
                DESCRIPTION "AMD rocCV is an efficient GPU-accelerated library for pre and post-processing of images using AMD’s HIP platform. \
rocCV develop package provides rocCV header files, samples, and license.txt")

cpack_add_component(asan
                DISPLAY_NAME "rocCV ASAN Package"
                DESCRIPTION "AMD rocCV is an efficient GPU-accelerated library for pre and post-processing of images using AMD’s HIP platform. \
rocCV ASAN package provides rocCV ASAN libraries")

cpack_add_component(test
                DISPLAY_NAME "rocCV Test Package"
                DESCRIPTION "AMD rocCV is an efficient GPU-accelerated library for pre and post-processing of images using AMD’s HIP platform. \
rocCV Test package provides rocCV Test Components")

unset(TESTS CACHE)
unset(FULL_BUILD CACHE)

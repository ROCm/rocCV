# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.24)


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
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# NOTE: rocCV version updates requires updates in
#       * rocCV source CMakeLists.txt
#       * python CMakeLists.txt
set(VERSION "0.4.0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Project Version and Language
project(roccv VERSION ${VERSION} LANGUAGES CXX C)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

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
option(SAMPLES                  "Build with samples"                            OFF)
option(TESTS                    "Build with tests"                              OFF)
option(BENCHMARKS               "Build with benchmarks"                         OFF)
option(ENABLE_CODE_COVERAGE      "Enable LLVM code coverage"                    OFF)
set(PYTHON_VERSION_SUGGESTED    ""          CACHE STRING "Python version to build rocCV")
set(DEFAULT_BUILD_TYPE          "Release"   CACHE STRING "Default build type for rocCV")

if(ENHANCED_MESSAGE)
	string(ASCII 27 Esc)
	set(ColorReset "${Esc}[m")
	set(Red "${Esc}[31m")
	set(Green "${Esc}[32m")
	set(Yellow "${Esc}[33m")
	set(Blue "${Esc}[34m")
	set(BoldBlue "${Esc}[1;34m")
	set(Magenta "${Esc}[35m")
	set(Cyan "${Esc}[36m")
	set(White "${Esc}[37m")
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Version -- ${VERSION}${ColorReset}")
message("-- ${BoldBlue}${PROJECT_NAME} Install Path -- ${CMAKE_INSTALL_PREFIX}${ColorReset}")

# rocCV Build Type
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE} CACHE STRING "rocCV Default Build Type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
	# -O0 		-- Don't Optimize output file
	# -g  		-- generate debugging information --- dwarf-4 for making valgrind work
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
else()
	# -O3       -- Optimize output file
	# -DNDEBUG  -- turn off asserts
	# -fPIC     -- Generate position-independent code if possible
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}${PROJECT_NAME} Build Type -- ${CMAKE_BUILD_TYPE}${ColorReset}")

# Set all warnings
# -Wall                -- Print all warnings
# -Wextra              -- Print extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Enable LLVM code coverage if build type is Debug and ENABLE_CODE_COVERAGE is set.
if(ENABLE_CODE_COVERAGE AND CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
elseif(ENABLE_CODE_COVERAGE AND NOT CMAKE_BUILD_TYPE MATCHES Debug)
    message("-- ${Yellow}LLVM code coverage enabled, but build type is not Debug. Code coverage will be disabled.${ColorReset}")
    set(ENABLE_CODE_COVERAGE OFF)
endif()

# Set all build flags if FULL_BUILD is set.
if(FULL_BUILD)
    set(SAMPLES ON)
    set(TESTS ON)
endif()

# Print developer options.
message("-- ${Cyan}${PROJECT_NAME} Developer Options${ColorReset}")
message("-- ${Cyan}     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} [Select rocCV build type [options:Debug/Release](default:Release)]${ColorReset}")
message("-- ${Cyan}     -D FULL_BUILD=${FULL_BUILD} [Compile with samples, and tests [options:ON/OFF](default:OFF)]${ColorReset}")
message("-- ${Cyan}     -D SAMPLES=${SAMPLES} [Compile with samples [options:ON/OFF](default:OFF)]${ColorReset}")
message("-- ${Cyan}     -D TESTS=${TESTS} [Compile with tests [options:ON/OFF](default:OFF)]${ColorReset}")
message("-- ${Cyan}     -D BENCHMARKS=${BENCHMARKS} [Compile with benchmarks [options:ON/OFF](default:OFF)]${ColorReset}")
message("-- ${Cyan}     -D BUILD_PYPACKAGE=${BUILD_PYPACKAGE} [Turn ON/OFF Build for rocCV Python Package(default:ON)]${ColorReset}")
message("-- ${Cyan}     -D PYTHON_VERSION_SUGGESTED=${PYTHON_VERSION_SUGGESTED} [User provided python version to use for rocCV Python Bindings(default:System Version)]${ColorReset}")
message("-- ${Cyan}     -D ENABLE_CODE_COVERAGE=${ENABLE_CODE_COVERAGE} [Enable LLVM code coverage [options:ON/OFF](default:OFF)]${ColorReset}")

# Set supported GPU Targets
if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
    find_package(ROCmCMakeBuildTools QUIET)
    if(NOT ROCmCMakeBuildTools_FOUND)
        find_package(ROCM QUIET)
    endif()
    include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

    set(OPTIONAL_GPU_TARGETS "gfx950;gfx1151;gfx1200;gfx1201")
    if(HAS_ROCM_CHECK_TARGET_IDS)
        rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
    else() # if we don't have rocm_check_target_ids, just assume the targets are available
        set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
    endif()
    set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
endif()

# Set AMD GPU_TARGETS
if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
    message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColorReset}")
    if(DEFINED ENV{AMDGPU_TARGETS})
        set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
    else()
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
endif()

if(DEFINED ENV{GPU_ARCHS})
    set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
elseif(GPU_TARGETS)
    message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColorReset}")
else()
    set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColorReset}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

# Determine if we are building using the TheRock install of ROCm.
set(USING_THE_ROCK OFF)
if (EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    set(USING_THE_ROCK ON)
endif()
message("-- ${White}${PROJECT_NAME} -- USING_THE_ROCK: ${USING_THE_ROCK}${ColorReset}")

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

# --- Find dependencies ---
find_package(HIP QUIET)
find_package(StdFilesystem QUIET)
find_package(OpenMP COMPONENTS CXX QUIET)
find_package(OpenCV QUIET)

# --- Dependency checks ---
set(BUILD_ROCCV true)
# HIP
if(NOT HIP_FOUND)
    set(BUILD_ROCCV false)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires HIP, Not Found${ColorReset}")
endif()

# OpenCV
if(NOT OpenCV_FOUND)
    if (SAMPLES)
        message("-- ${Yellow}NOTE: ${PROJECT_NAME} building samples requires OpenCV, Not Found ${ColorReset}")
        set(SAMPLES OFF)
    endif()
    set(FULL_BUILD OFF)
endif()
# OpenMP
if (NOT OpenMP_CXX_FOUND)
    set(BUILD_ROCCV false)
    message("-- ${Yellow}NOTE: ${PROJECT_NAME} library requires OpenMP, Not Found${ColorReset}")
endif()

# Check that all dependencies have been met
if(NOT BUILD_ROCCV)
	message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} dependencies not satisfied${ColorReset}")
endif()

set(ROCCV_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# --- Source code ---
add_subdirectory(src)

# --- Samples ---
if(SAMPLES)
    add_subdirectory(samples)
endif(SAMPLES)

# --- Benchmarks ---
if(BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BENCHMARKS)

# --- Tests ---
if(TESTS)
    include(CTest)
    enable_testing()
	add_subdirectory(tests)
endif(TESTS)

# --- Python package ---
if(BUILD_PYPACKAGE)
    add_subdirectory("python")
endif(BUILD_PYPACKAGE)

# --- Mark test files for installation ---

# Pybind tests - Mark for installation
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

# CPP Test - Mark for installation
install(DIRECTORY ${ROCCV_SOURCE_DIR}/data
        DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test
        COMPONENT test
)
install(DIRECTORY ${ROCCV_SOURCE_DIR}/tests/roccv/cpp
        DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test
        COMPONENT test
)

# Samples - Mark for installation
install(DIRECTORY ${ROCCV_SOURCE_DIR}/samples
        DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv
        COMPONENT dev
)

# --- Packaging configuration ---
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

# set license information - {ROCM_PATH}/share/doc/roccv
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
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
set(ROCCV_RUNTIME_PACKAGE_LIST  "hip-runtime-amd")
# * Dev
set(ROCCV_DEBIAN_DEV_PACKAGE_LIST  "hip-dev, python3-dev")
set(ROCCV_RPM_DEV_PACKAGE_LIST  "hip-devel, python3-devel")
# * Test
set(ROCCV_DEBIAN_TEST_PACKAGE_LIST  "cmake, python3-pytest, python3-numpy")
set(ROCCV_RPM_TEST_PACKAGE_LIST  "cmake, python3-pytest, python3-numpy")

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
    message("-- ${White}AMD ROCm rocCV rpm RunTime Package -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColorReset}")
    message("-- ${White}AMD ROCm rocCV rpm Dev Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColorReset}")
    message("-- ${White}AMD ROCm rocCV rpm Test Package -- ${CPACK_RPM_TEST_PACKAGE_REQUIRES}${ColorReset}")
endif()
if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}AMD ROCm rocCV deb RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColorReset}")
    message("-- ${White}AMD ROCm rocCV deb Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColorReset}")
    message("-- ${White}AMD ROCm rocCV deb Test Package -- ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS}${ColorReset}")
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

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

file(GLOB_RECURSE ROCCV_SRC "*.cpp")
add_library(${PROJECT_NAME} SHARED ${ROCCV_SRC})
add_library(roccv::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

target_compile_options(${PROJECT_NAME}
    PUBLIC
        ${OpenMP_CXX_FLAGS}
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        hip::device
        OpenMP::OpenMP_CXX
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${ROCCV_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/roccv>
)
# Require C++20 so downstream gets the right standard when linking to roccv::roccv
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    BUILD_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib/llvm/lib"
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/rocm_sysdeps/lib;\$ORIGIN/llvm/lib"
)

# Add export set

# install rocCV libs -- {ROCM_PATH)/lib
install(TARGETS ${PROJECT_NAME}
    EXPORT roccvTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT runtime
        NAMELINK_SKIP
)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT dev
        NAMELINK_ONLY
)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

# install roccvTargets.cmake
install(EXPORT roccvTargets
    FILE roccvTargets.cmake
    NAMESPACE roccv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/roccv
    COMPONENT dev
)

# install roccvConfig.cmake and roccvConfigVersion.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${ROCCV_SOURCE_DIR}/cmake/roccvConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/roccvConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/roccv
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/roccvConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/roccvConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/roccvConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/roccv
    COMPONENT dev
)

# install rocCV include files -- {ROCM_PATH}/include/roccv/
install(DIRECTORY ${ROCCV_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roccv COMPONENT dev)

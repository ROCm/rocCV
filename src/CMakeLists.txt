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

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
        ${ROCCV_SOURCE_DIR}/include
)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib;${ROCM_PATH}/lib/llvm/lib" BUILD_WITH_INSTALL_RPATH TRUE)

# install rocCV libs -- {ROCM_PATH)/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install rocCV include files -- {ROCM_PATH}/include/roccv/
install(DIRECTORY ${ROCCV_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roccv COMPONENT dev)
# CPP Test - Installed
install(DIRECTORY ${ROCCV_SOURCE_DIR}/data DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test COMPONENT test)
install(DIRECTORY ${ROCCV_SOURCE_DIR}/tests/roccv/cpp DESTINATION ${CMAKE_INSTALL_DATADIR}/roccv/test COMPONENT test)

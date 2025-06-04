/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

namespace roccv {

/**
 * @brief Describes various status types for roccv Exceptions
 *
 */
enum class eStatusType {
    SUCCESS,
    INVALID_HANDLE,      /** @brief handle parameter is invalid. */
    INVALID_POINTER,     /** @brief pointer parameter is invalid (e.g., nullptr). */
    INVALID_VALUE,       /** @brief value of parameter is invalid. */
    OUT_OF_BOUNDS,       /** @brief numerical value of parameter is out of the allowed
                            range. */
    OUT_OF_MEMORY,       /** @brief execution cannot be completed due to lack of
                          * memory.
                          */
    INVALID_OPERATION,   /** @brief the operation cannot be executed because of
                            incorrect context. */
    INTERNAL_ERROR,      /** @brief an error occured during the execution. If a point
                            that shouldn't be reached is reached. */
    INVALID_COMBINATION, /** @brief invalid permutation of parameters. */
    NOT_IMPLEMENTED,     /** @brief cannot execute because the current permutation
                            has not been implemented. */
    NOT_INITIALIZED,     /** @brief object parameter has not been inititalized. */
};

/**
 * @brief Describes various status types for roccv test Exceptions. It may be worth moving this into the standalone test
 * folder in the future, as it has no use in the library itself.
 *
 */
enum class eTestStatusType {
    TEST_SUCCESS,
    TEST_FAILURE,
    UNEXPECTED_VALUE, /** @brief the value does not match with the expected
                         value. */
};
}  // namespace roccv
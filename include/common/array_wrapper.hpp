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

#include <hip/hip_runtime.h>

/**
 * @brief A helper class to transfer fixed size arrays to device.
 *
 * @tparam T the data type of the array
 * @tparam N the size of the array
 */
template <typename T, size_t N>
class ArrayWrapper {
   public:
    /**
     * @brief Construct a new Array Wrapper object
     *
     * @param[in] data the data to be used for the ArrayWrapper
     */
    ArrayWrapper(const T *data) {
        if (data == nullptr) {
            return;
        }

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            this->data_[i] = data[i];
        }
    }

    __device__ __host__ T &operator[](size_t i) { return this->data_[i]; }

    __device__ __host__ const T &operator[](size_t i) const {
        return this->data_[i];
    }

   private:
    T data_[N];
};
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

#include <hip/hip_vector_types.h>
#include <hip/math_functions.h>

#include <type_traits>

class MathVector {
   public:
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 1), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, rank> pow(
        HIP_vector_type<T, rank> vec, double p) {
        vec.x = ::pow(vec.x, p);
        return vec;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 2), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, rank> pow(
        HIP_vector_type<T, rank> vec, double p) {
        vec.x = ::pow(vec.x, p);
        vec.y = ::pow(vec.y, p);
        return vec;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 3), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, rank> pow(
        HIP_vector_type<T, rank> vec, double p) {
        vec.x = ::pow(vec.x, p);
        vec.y = ::pow(vec.y, p);
        vec.z = ::pow(vec.z, p);
        return vec;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 4), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, rank> pow(
        HIP_vector_type<T, rank> vec, double p) {
        vec.x = ::pow(vec.x, p);
        vec.y = ::pow(vec.y, p);
        vec.z = ::pow(vec.z, p);
        vec.w = ::pow(vec.w, p);
        return vec;
    }

    template <typename U, typename T, unsigned int rank,
              typename std::enable_if<(rank == 1), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<U, rank> convert_base(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<U, rank> out;
        out.x = static_cast<U>(vec.x);
        return out;
    }
    template <typename U, typename T, unsigned int rank,
              typename std::enable_if<(rank == 2), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<U, rank> convert_base(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<U, rank> out;
        out.x = static_cast<U>(vec.x);
        out.y = static_cast<U>(vec.y);
        return out;
    }
    template <typename U, typename T, unsigned int rank,
              typename std::enable_if<(rank == 3), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<U, rank> convert_base(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<U, rank> out;
        out.x = static_cast<U>(vec.x);
        out.y = static_cast<U>(vec.y);
        out.z = static_cast<U>(vec.z);
        return out;
    }
    template <typename U, typename T, unsigned int rank,
              typename std::enable_if<(rank == 4), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<U, rank> convert_base(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<U, rank> out;
        out.x = static_cast<U>(vec.x);
        out.y = static_cast<U>(vec.y);
        out.z = static_cast<U>(vec.z);
        out.w = static_cast<U>(vec.w);
        return out;
    }

    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 1), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, 4> fill(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<T, 4> out{vec.x, 0, 0, 0};
        return out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 2), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, 4> fill(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<T, 4> out{vec.x, vec.y, 0, 0};
        return out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 3), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, 4> fill(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<T, 4> out{vec.x, vec.y, vec.z, 0};
        return out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 4), bool>::type = 0>
    __host__ __device__ static HIP_vector_type<T, 4> fill(
        HIP_vector_type<T, rank> vec) {
        HIP_vector_type<T, 4> out{vec.x, vec.y, vec.z, vec.w};
        return out;
    }

    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 1), bool>::type = 0>
    __host__ __device__ static void trunc(HIP_vector_type<T, 4> vec,
                                          HIP_vector_type<T, rank> *dst) {
        HIP_vector_type<T, rank> out{vec.x /*, vec.y, vec.z, vec.w*/};
        *dst = out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 2), bool>::type = 0>
    __host__ __device__ static void trunc(HIP_vector_type<T, 4> vec,
                                          HIP_vector_type<T, rank> *dst) {
        HIP_vector_type<T, rank> out{vec.x, vec.y /*, vec.z, vec.w*/};
        *dst = out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 3), bool>::type = 0>
    __host__ __device__ static void trunc(HIP_vector_type<T, 4> vec,
                                          HIP_vector_type<T, rank> *dst) {
        HIP_vector_type<T, rank> out{vec.x, vec.y, vec.z /*, vec.w*/};
        *dst = out;
    }
    template <typename T, unsigned int rank,
              typename std::enable_if<(rank == 4), bool>::type = 0>
    __host__ __device__ static void trunc(HIP_vector_type<T, 4> vec,
                                          HIP_vector_type<T, rank> *dst) {
        HIP_vector_type<T, rank> out{vec.x, vec.y, vec.z, vec.w};
        *dst = out;
    }
};

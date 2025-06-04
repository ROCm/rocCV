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

#include <algorithm>
#include <vector>

#include "core/tensor.hpp"

/**
 * @brief Validates whether a tensor is located on a specified device.
 *
 */
#define CHECK_TENSOR_DEVICE(tensor, tensor_device)                                             \
    if (tensor.device() != tensor_device) {                                                    \
        throw roccv::Exception("Invalid tensor " #tensor                                       \
                               ": Ensure that this tensor is allocated on the proper device.", \
                               eStatusType::INVALID_OPERATION);                                \
    }

/**
 * @brief Validates whether a tensor's layout is supported, based on a list of
 * provided supported layouts.
 *
 */
#define CHECK_TENSOR_LAYOUT(tensor, ...)                                                                     \
    do {                                                                                                     \
        std::vector<eTensorLayout> v{__VA_ARGS__};                                                           \
        if (std::find(v.begin(), v.end(), tensor.layout().elayout()) == v.end()) {                           \
            throw roccv::Exception("Unsupported tensor layout: " #tensor, eStatusType::INVALID_COMBINATION); \
        }                                                                                                    \
    } while (0);

/**
 * @brief Validates whether a tensor's datatype is supported based on a list of
 * provided datatypes.
 *
 */
#define CHECK_TENSOR_DATATYPES(tensor, ...)                                                          \
    do {                                                                                             \
        std::vector<eDataType> v{__VA_ARGS__};                                                       \
        if (std::find(v.begin(), v.end(), tensor.dtype().etype()) == v.end()) {                      \
            throw roccv::Exception("Unsupported data type: " #tensor, eStatusType::NOT_IMPLEMENTED); \
        }                                                                                            \
    } while (0);

/**
 * @brief Used for generic tensor validation comparisons. For example:
 * (input.layout() == output.layout()).
 *
 */
#define CHECK_TENSOR_COMPARISON(comparison)                                                            \
    if (!(comparison)) {                                                                               \
        throw roccv::Exception("Tensor check failed: " #comparison, eStatusType::INVALID_COMBINATION); \
    }

/**
 * @brief Validates whether a tensor contains a supported number of channels.
 *
 */
#define CHECK_TENSOR_CHANNELS(tensor, ...)                                                                   \
    do {                                                                                                     \
        const std::vector<unsigned int> v{__VA_ARGS__};                                                      \
        if (std::find(v.begin(), v.end(), tensor.shape(tensor.layout().channels_index())) == v.end()) {      \
            throw roccv::Exception("Unsupported channel count: " #tensor, eStatusType::INVALID_COMBINATION); \
        }                                                                                                    \
    } while (0);
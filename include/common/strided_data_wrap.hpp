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

#include <cstddef>
#include <iostream>
#include <type_traits>

#include "core/exception.hpp"
#include "core/status_type.h"
#include "core/tensor.hpp"
#include "core/tensor_data.hpp"

namespace roccv::detail {

template </*typename T, */ size_t RANK>
class StridedDataWrap {
   public:
    template <typename... ARGS>
    StridedDataWrap(void *data, ARGS... strides)
        : data_(static_cast<uchar *>(data)), strides_{strides...} {
        static_assert(sizeof...(ARGS) == RANK,
                      "StridedDataWrap sizeof...(ARGS) == RANK");
    }

    template <typename T, typename... ARGS>
    __device__ __host__ T &at(ARGS... idx) {
        return const_cast<T &>(
            static_cast<const StridedDataWrap<RANK> &>(*this).at<T>(
                std::forward<ARGS>(idx)...));
    }

    template <typename T, typename... ARGS>
    __device__ __host__ const T &at(ARGS... idx) const {
        static_assert(sizeof...(ARGS) == RANK,
                      "StridedDataWrap sizeof...(ARGS) == RANK");

        int64_t coords[] = {idx...};

        size_t index = 0;
#pragma unroll
        for (int i = 0; i < RANK; ++i) {
            index += coords[i] * strides_[i];
        }

        return *(static_cast<T *>(static_cast<void *>(this->data_ + index)));
    }

   private:
    unsigned char *data_;
    int64_t strides_[RANK];
};

template <eTensorLayout LAYOUT>
constexpr size_t layout_get_rank();

template <>
constexpr size_t layout_get_rank<TENSOR_LAYOUT_LNHWC>() {
    return 5;
}

template <>
constexpr size_t layout_get_rank<TENSOR_LAYOUT_NHWC>() {
    return 4;
}

template <>
constexpr size_t layout_get_rank<TENSOR_LAYOUT_HWC>() {
    return 3;
}

template <>
constexpr size_t layout_get_rank<TENSOR_LAYOUT_NMC>() {
    return 3;
}

template <>
constexpr size_t layout_get_rank<TENSOR_LAYOUT_NMD>() {
    return 3;
}

template <eTensorLayout LAYOUT>
inline auto get_sdwrapper(const roccv::Tensor &tensor)
    -> const StridedDataWrap<layout_get_rank<LAYOUT>()>;

template <eTensorLayout LAYOUT>
inline auto get_sdwrapper(roccv::Tensor &tensor)
    -> StridedDataWrap<layout_get_rank<LAYOUT>()> {
    return get_sdwrapper<LAYOUT>(static_cast<const roccv::Tensor &>(tensor));
}

template <>
inline auto get_sdwrapper<TENSOR_LAYOUT_NHWC>(const roccv::Tensor &tensor)
    -> const StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NHWC>()> {
    auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();

    if (tensor.shape().layout() != TENSOR_LAYOUT_NHWC &&
        tensor.shape().layout() != TENSOR_LAYOUT_HWC) {
        throw Exception(eStatusType::INTERNAL_ERROR);
    }

    const auto height_s =
        tensor_data.stride(tensor.shape().layout().height_index());
    const auto width_s =
        tensor_data.stride(tensor.shape().layout().width_index());
    const auto channels_s =
        tensor_data.stride(tensor.shape().layout().channels_index());

    const auto batch_i = tensor.shape().layout().batch_index();
    const auto batch_s =
        (batch_i >= 0)
            ? tensor_data.stride(batch_i)
            : height_s * tensor.shape()[tensor.shape().layout().height_index()];

    return StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NHWC>()>(
        tensor_data.basePtr(), batch_s, height_s, width_s, channels_s);
}

template <>
inline auto get_sdwrapper<TENSOR_LAYOUT_LNHWC>(const roccv::Tensor &tensor)
    -> const StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_LNHWC>()> {
    auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();

    if (tensor.shape().layout() != TENSOR_LAYOUT_LNHWC) {
        throw Exception(eStatusType::INTERNAL_ERROR);
    }

    const auto height_s =
        tensor_data.stride(tensor.shape().layout().height_index());
    const auto width_s =
        tensor_data.stride(tensor.shape().layout().width_index());
    const auto channels_s =
        tensor_data.stride(tensor.shape().layout().channels_index());

    const auto batch_i = tensor.shape().layout().batch_index();
    const auto batch_s = tensor_data.stride(batch_i);

    const auto layer_s =
        tensor_data.stride(tensor.shape().layout().sift_octave_layer_index());

    return StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_LNHWC>()>(
        tensor_data.basePtr(), layer_s, batch_s, height_s, width_s, channels_s);
}

template <>
inline auto get_sdwrapper<TENSOR_LAYOUT_NMC>(const roccv::Tensor &tensor)
    -> const StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NMC>()> {
    auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();

    if (tensor.shape().layout() != TENSOR_LAYOUT_NMC) {
        throw Exception(eStatusType::INTERNAL_ERROR);
    }
    const auto batch_i = tensor.shape().layout().batch_index();
    const auto batch_s = tensor_data.stride(batch_i);
    const auto maxFeatures_s =
        tensor_data.stride(tensor.shape().layout().max_features_index());
    const auto features_s =
        tensor_data.stride(tensor.shape().layout().sift_features_index());

    return StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NMC>()>(
        tensor_data.basePtr(), batch_s, maxFeatures_s, features_s);
}

template <>
inline auto get_sdwrapper<TENSOR_LAYOUT_NMD>(const roccv::Tensor &tensor)
    -> const StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NMD>()> {
    auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();

    if (tensor.shape().layout() != TENSOR_LAYOUT_NMD) {
        throw Exception(eStatusType::INTERNAL_ERROR);
    }
    const auto batch_i = tensor.shape().layout().batch_index();
    const auto batch_s = tensor_data.stride(batch_i);
    const auto maxFeatures_s =
        tensor_data.stride(tensor.shape().layout().max_features_index());
    const auto features_s =
        tensor_data.stride(tensor.shape().layout().sift_features_index());

    return StridedDataWrap<layout_get_rank<TENSOR_LAYOUT_NMD>()>(
        tensor_data.basePtr(), batch_s, maxFeatures_s, features_s);
}
}
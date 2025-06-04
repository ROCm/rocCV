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

#include <stdint.h>

#include <unordered_map>

#include "exception.hpp"
#include "util_enums.h"

/**
 * @brief The max number of dimensions of a rocCV tensor.
 */
#define ROCCV_TENSOR_MAX_RANK (15)

namespace roccv {
/**
 * @brief Descriptors used to specify features of a specific tensor layout type.
 *
 */
struct TensorLayoutDesc {
    int32_t rank;
    int32_t batch_index;
    int32_t width_index;
    int32_t height_index;
    int32_t channel_index;
    int32_t max_features_index;
    int32_t sift_features_index;
    int32_t sift_octave_layer_index;
};

/**
 * @brief TensorLayout class.
 *
 */
class TensorLayout {
   public:
    /**
     * @brief Construct a new Tensor Layout object
     *
     * @param[in] layout The desired layout of the TensorLayout object. See
     * eTensorLayout for information on supported layouts.
     */
    explicit TensorLayout(eTensorLayout layout) {
        if (TensorLayout::layoutDescriptorTable.count(layout) == 0) {
            throw Exception("Invalid TensorLayout type", eStatusType::INVALID_VALUE);
        }

        layout_ = layout;
        layout_desc_ = TensorLayout::layoutDescriptorTable.at(layout);
    }

    /**
     * @brief Provides descriptors for each feature of a specified layout type.
     */
    inline static const std::unordered_map<eTensorLayout, TensorLayoutDesc> layoutDescriptorTable = {
        {TENSOR_LAYOUT_HWC, {3, -1, 1, 0, 2, -1, -1, -1}}, {TENSOR_LAYOUT_NC, {2, 0, -1, -1, 1, -1, -1, -1}},
        {TENSOR_LAYOUT_NW, {2, 0, 1, -1, -1, -1, -1, -1}}, {TENSOR_LAYOUT_NHWC, {4, 0, 2, 1, 3, -1, -1, -1}},
        {TENSOR_LAYOUT_NMC, {3, 0, -1, -1, -1, 1, 2, -1}}, {TENSOR_LAYOUT_NMD, {3, 0, -1, -1, -1, 1, 2, -1}},
        {TENSOR_LAYOUT_LNHWC, {5, 1, 3, 2, 4, -1, -1, 0}}, {TENSOR_LAYOUT_NCHW, {4, 0, 3, 2, 1, -1, -1, -1}},
        {TENSOR_LAYOUT_N, {1, 0, -1, -1, -1, -1, -1, -1}}, {TENSOR_LAYOUT_NWC, {3, 0, 1, -1, 2, -1, -1, -1}}};

    /**
     * @brief Returns the layout enum stored in the TensorLayout object.
     *
     * @return eTensorLayout
     */
    eTensorLayout elayout() const { return layout_; }

    bool operator==(const eTensorLayout &rhs) const { return this->layout_ == rhs; }

    bool operator!=(const eTensorLayout &rhs) const { return !operator==(rhs); }

    bool operator==(const TensorLayout &rhs) const { return this->layout_ == rhs.layout_; }

    bool operator!=(const TensorLayout &rhs) const { return !operator==(rhs); }

    /**
     * @brief Returns the rank of the Tensor Layout object.
     *
     * @return int32_t
     */
    int32_t rank() const { return layout_desc_.rank; }

    /**
     * @brief Index of the batch dimension specified by layout. E.g. returns 0
     * for TENSOR_LAYOUT_NHWC.
     * @return Index or -1 if the layout does not have a batch dimension.
     */
    int32_t batch_index() const { return layout_desc_.batch_index; }

    /**
     * @brief Index of the height dimension specified by layout. E.g. returns 1
     * for TENSOR_LAYOUT_NHWC.
     * @return Index of the height dimension.
     */
    int32_t height_index() const { return layout_desc_.height_index; }

    /**
     * @brief Index of the width dimension specified by layout. E.g. returns 2
     * for TENSOR_LAYOUT_NHWC.
     * @return Index of the width dimension.
     */
    int32_t width_index() const { return layout_desc_.width_index; }

    /**
     * @brief Index of the channels dimension specified by layout. E.g. returns
     * 3 for TENSOR_LAYOUT_NHWC.
     * @return Index of the channels dimension.
     */
    int32_t channels_index() const { return layout_desc_.channel_index; }

    /**
     * @brief Index of the max features dimension specified by layout
     *
     * @return Index of the max features dimension or -1 if the layout does not
     * contain it.
     */
    int32_t max_features_index() const { return layout_desc_.max_features_index; }

    /**
     * @brief Index of the sift features dimension specified by layout
     *
     * @return int32_t
     */
    int32_t sift_features_index() const { return layout_desc_.sift_features_index; }

    /**
     * @brief Index of the sift octave layer dimension specified by layout
     *
     * @return int32_t
     */
    int32_t sift_octave_layer_index() const { return layout_desc_.sift_octave_layer_index; }

   private:
    eTensorLayout layout_;
    TensorLayoutDesc layout_desc_;
};
}  // namespace roccv
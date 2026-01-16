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
    explicit TensorLayout(eTensorLayout layout);

    // clang-format off
    inline static const std::unordered_map<eTensorLayout, std::string> layoutStringTable = {
        {TENSOR_LAYOUT_HWC,     "HWC"},
        {TENSOR_LAYOUT_NC,      "NC"},
        {TENSOR_LAYOUT_NW,      "NW"},
        {TENSOR_LAYOUT_NHWC,    "NHWC"},
        {TENSOR_LAYOUT_NMC,     "NMC"},
        {TENSOR_LAYOUT_NMD,     "NMD"},
        {TENSOR_LAYOUT_LNHWC,   "LNHWC"},
        {TENSOR_LAYOUT_NCHW,    "NCHW"},
        {TENSOR_LAYOUT_N,       "N"},
        {TENSOR_LAYOUT_NWC,     "NWC"},
    };
    // clang-format on

    /**
     * @brief Returns the index of the given dimension in the layout.
     *
     * @param[in] dimension The dimension to get the index of.
     * @return The index of the dimension, or -1 if the dimension is not found in the layout.
     */
    int32_t indexOf(std::string_view dim) const;

    /**
     * @brief Returns the layout string representing the layout.
     *
     * @return The layout string.
     */
    inline const std::string &string() const { return m_layoutString; }

    /**
     * @brief Returns the layout enum stored in the TensorLayout object.
     *
     * @return eTensorLayout
     */
    eTensorLayout elayout() const { return m_layout; }

    bool operator==(const eTensorLayout &rhs) const { return this->m_layout == rhs; }
    bool operator!=(const eTensorLayout &rhs) const { return !operator==(rhs); }
    bool operator==(const TensorLayout &rhs) const { return this->m_layout == rhs.m_layout; }
    bool operator!=(const TensorLayout &rhs) const { return !operator==(rhs); }

    /**
     * @brief Returns the rank of the Tensor Layout object.
     *
     * @return int32_t
     */
    int32_t rank() const { return m_rank; }

    /**
     * @brief Index of the batch dimension specified by layout. E.g. returns 0
     * for TENSOR_LAYOUT_NHWC.
     * @return Index or -1 if the layout does not have a batch dimension.
     */
    int32_t batch_index() const { return indexOf("N"); }

    /**
     * @brief Index of the height dimension specified by layout. E.g. returns 1
     * for TENSOR_LAYOUT_NHWC.
     * @return Index of the height dimension.
     */
    int32_t height_index() const { return indexOf("H"); }

    /**
     * @brief Index of the width dimension specified by layout. E.g. returns 2
     * for TENSOR_LAYOUT_NHWC.
     * @return Index of the width dimension.
     */
    int32_t width_index() const { return indexOf("W"); }

    /**
     * @brief Index of the channels dimension specified by layout. E.g. returns
     * 3 for TENSOR_LAYOUT_NHWC.
     * @return Index of the channels dimension.
     */
    int32_t channels_index() const { return indexOf("C"); }

   private:
    eTensorLayout m_layout;
    std::string m_layoutString;
    int m_rank;
};
}  // namespace roccv
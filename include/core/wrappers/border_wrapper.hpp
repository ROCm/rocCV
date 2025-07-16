/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "core/wrappers/image_wrapper.hpp"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Wrapper class for ImageWrapper. This extends the descriptors by defining behaviors for when tensor
 * coordinates go out of scope.
 *
 * @tparam T The underlying data type of the tensor.
 * @tparam BorderType The border type to use when coordinates are out of bounds.
 */
template <typename T, eBorderType BorderType>
class BorderWrapper {
   public:
    /**
     * @brief Wraps an ImageWrapper and extends its capabilities to handle out of bounds coordinates.
     *
     * @param tensor The tensor to wrap.
     * @param border_value The fallback border color to use when using a constant border mode.
     */
    BorderWrapper(const Tensor& tensor, T border_value) : m_desc(tensor), m_border_value(border_value) {}

    /**
     * @brief Constructs a BorderWrapper from an existing ImageWrapper. Extends its capabilities to handle out of bound
     * coordinates.
     *
     * @param image_wrapper The ImageWrapper to wrap around the BorderWrapper.
     * @param border_value The fallback border color to use when using a constant border mode.
     */
    BorderWrapper(ImageWrapper<T> image_wrapper, T border_value)
        : m_desc(image_wrapper), m_border_value(border_value) {}

    /**
     * @brief Returns a reference to the underlying data given image coordinates. If the coordinates fall out of bounds,
     * a fallback reference based on the provided border type will be given instead.
     *
     * @param n The batch index.
     * @param h The height index.
     * @param w The width index.
     * @param c The channel index.
     * @return A reference to the underlying data or a fallback border value of type T.
     */
    __device__ __host__ const T at(int64_t n, int64_t h, int64_t w, int64_t c) const {
        // Constant border type implementation
        if constexpr (BorderType == eBorderType::BORDER_TYPE_CONSTANT) {
            if (w < 0 || w >= width() || h < 0 || h >= height())
                return m_border_value;
            else
                return m_desc.at(n, h, w, c);
        }

        // Reflect border type implementation
        else if constexpr (BorderType == eBorderType::BORDER_TYPE_REFLECT) {
            // clang-format off
            bool x_overflow_n = (w < 0);
            bool x_overflow_p = (w >= width());
            int64_t x = (x_overflow_n) * (std::abs(w + 1) % width()) +
                        (x_overflow_p) * (width() - (w % width()) - 1) + 
                        (!x_overflow_n && !x_overflow_p) * w;
            
        
            bool y_overflow_n = (h < 0);
            bool y_overflow_p = (h >= height());
            int64_t y = (y_overflow_n) * (std::abs(h + 1) % height()) +
                        (y_overflow_p) * (height() - (h % height()) - 1) + 
                        (!y_overflow_n && !y_overflow_p) * h;
            // clang-format on

            return m_desc.at(n, y, x, c);
        }

        // Replicate border type implementation
        else if constexpr (BorderType == eBorderType::BORDER_TYPE_REPLICATE) {
            int64_t x = std::clamp<int64_t>(w, 0, width() - 1);
            int64_t y = std::clamp<int64_t>(h, 0, height() - 1);
            return m_desc.at(n, y, x, c);
        }

        // Wrap border type implementation
        else if constexpr (BorderType == eBorderType::BORDER_TYPE_WRAP) {
            // clang-format off
            bool x_overflow = (w < 0 || w >= width());
            int64_t x = x_overflow * (w % width() + width()) % width() +
                        !x_overflow * w;

            bool y_overflow = (h < 0 || h >= height());
            int64_t y = y_overflow * (h % height() + height()) % height() +
                        !y_overflow * h;
            //clang-format on

            return m_desc.at(n, y, x, c);
        }

        else {
            static_assert(false, "BorderType tparam must be a supported border mode.");
        }
    }

    /**
     * @brief Retrives the height of the images.
     *
     * @return Image height.
     */
    __device__ __host__ inline int64_t height() const { return m_desc.height(); }

    /**
     * @brief Retrieves the width of the image.
     *
     * @return Image width.
     */
    __device__ __host__ inline int64_t width() const { return m_desc.width(); }

    /**
     * @brief Retrieves the number of batches in the image tensor.
     *
     * @return Number of batches.
     */
    __device__ __host__ inline int64_t batches() const { return m_desc.batches(); }

    /**
     * @brief Retries the number of channels in the image.
     *
     * @return Image channels.
     */
    __device__ __host__ inline int64_t channels() const { return m_desc.channels(); }

   private:
    ImageWrapper<T> m_desc;
    T m_border_value;
};
}  // namespace roccv
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
        // Constant border type implementation. This is a special case which doesn't remap values, but rather returns
        // the provided constant value.
        if constexpr (BorderType == eBorderType::BORDER_TYPE_CONSTANT) {
            if (w < 0 || w >= width() || h < 0 || h >= height())
                return m_border_value;
            else
                return m_desc.at(n, h, w, c);
        }

        // We can return early if our coordinates are within the bounds. This is to avoid expensive calculations
        // required at image borders. While this may cause branch divergence, a good bulk of the pixels should fall
        // within image bounds and will take the same branch. This is preferred over having to do expensive calculations
        // for EVERY pixel in the image (most of which do not require said calculations).
        if (w >= 0 && w < width() && h >= 0 && h < height()) {
            return m_desc.at(n, h, w, c);
        }

        // Otherwise, do some additional calculations to map the provided x and y coordinates to be within bounds.
        int64_t x = w, y = h;

        // Reflect border type implementation. (Note: This is NOT REFLECT101, pixels at the border will be duplicated as
        // is the intended behavior for this border mode.)
        if constexpr (BorderType == eBorderType::BORDER_TYPE_REFLECT) {
            int64_t tWidth = width();
            // There is a special case if we have a dimension of size 1
            if (tWidth == 1) {
                x = 0;
            } else {
                int64_t scale = tWidth * 2;
                int64_t val = (w % scale + scale) % scale;
                x = (val < tWidth) ? val : scale - 1 - val;
            }

            int64_t tHeight = height();
            if (tHeight == 1) {
                y = 0;
            } else {
                int64_t scale = tHeight * 2;
                int64_t val = (h % scale + scale) % scale;
                y = (val < tHeight) ? val : scale - 1 - val;
            }
        }

        // Replicate border type implementation
        if constexpr (BorderType == eBorderType::BORDER_TYPE_REPLICATE) {
            x = std::clamp<int64_t>(w, 0, width() - 1);
            y = std::clamp<int64_t>(h, 0, height() - 1);
        }

        // Wrap border type implementation
        if constexpr (BorderType == eBorderType::BORDER_TYPE_WRAP) {
            if (w < 0 || w >= width()) {
                x = (w % width() + width()) % width();
            }

            if (h < 0 || h >= height()) {
                y = (h % height() + height()) % height();
            }
        }

        return m_desc.at(n, y, x, c);
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
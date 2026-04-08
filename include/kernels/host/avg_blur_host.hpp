/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/vector_utils.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Host {

/**
 * @brief CPU implementation of 2D Average Blur
 *
 * Applies a 2D averaging filter to the input image on CPU. Each output pixel is computed
 * as the average of all pixels within the kernel window.
 *
 * @tparam T Data type (e.g., uchar3, float4, etc.)
 * @tparam SrcWrapper Border wrapper type for input
 * @tparam DstWrapper Image wrapper type for output
 *
 * @param input Input image with border handling
 * @param output Output image
 * @param kernelWidth Width of the averaging kernel
 * @param kernelHeight Height of the averaging kernel
 * @param kernelAnchorX X-coordinate of kernel anchor point
 * @param kernelAnchorY Y-coordinate of kernel anchor point
 */
template <typename T, typename SrcWrapper, typename DstWrapper>
void avg_blur_2d(SrcWrapper input, DstWrapper output,
                 int kernelWidth, int kernelHeight,
                 int kernelAnchorX, int kernelAnchorY) {
    using namespace roccv::detail;
    using WorkType = MakeType<float, NumElements<T>>;

    // Compute kernel area for averaging
    float kernelArea = static_cast<float>(kernelWidth * kernelHeight);

    // Iterate over all batches
    for (int b = 0; b < output.batches(); b++) {
        // Iterate over all output pixels
        for (int y = 0; y < output.height(); y++) {
            for (int x = 0; x < output.width(); x++) {
                // Initialize accumulator
                WorkType sum = SetAll<WorkType>(0.0f);

                // Compute the sum over the kernel window
                for (int ky = 0; ky < kernelHeight; ++ky) {
                    int srcY = y - kernelAnchorY + ky;

                    for (int kx = 0; kx < kernelWidth; ++kx) {
                        int srcX = x - kernelAnchorX + kx;

                        // Read pixel through border wrapper (handles out-of-bounds)
                        T pixel = input.at(b, srcY, srcX, 0);

                        // Accumulate as float to avoid overflow
                        sum = sum + StaticCast<WorkType>(pixel);
                    }
                }

                // Compute average by dividing by kernel area
                WorkType average = sum / kernelArea;

                // Write result with saturation
                output.at(b, y, x, 0) = SaturateCast<T>(average);
            }
        }
    }
}

}  // namespace Host
}  // namespace Kernels

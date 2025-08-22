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

#include "core/wrappers/generic_tensor_wrapper.hpp"
#include "kernels/common/non_max_suppression_helpers.hpp"

namespace Kernels {
namespace Host {
inline void non_maximum_suppression(roccv::GenericTensorWrapper<short4> input,
                                    roccv::GenericTensorWrapper<uint8_t> output,
                                    roccv::GenericTensorWrapper<float> scores, int numBoxes, float scoresThreshold,
                                    float iouThreshold) {
    for (int64_t batchIdx = 0; batchIdx < input.shape[0]; batchIdx++) {
        for (int64_t boxAIdx = 0; boxAIdx < numBoxes; boxAIdx++) {
            const float scoreA = scores.at(batchIdx, boxAIdx);
            uint8_t& dst = output.at(batchIdx, boxAIdx);

            if (scoreA < scoresThreshold) {
                dst = 0u;
                continue;
            }

            const short4 boxA = input.at(batchIdx, boxAIdx);
            bool discard = false;

            for (int boxBIdx = 0; boxBIdx < numBoxes; boxBIdx++) {
                if (boxBIdx == boxAIdx) continue;

                const short4 boxB = input.at(batchIdx, boxBIdx);
                if (ComputeIoU(boxA, boxB) > iouThreshold) {
                    const float scoreB = scores.at(batchIdx, boxBIdx);
                    if (scoreA < scoreB || (scoreA == scoreB && ComputeArea(boxA) < ComputeArea(boxB))) {
                        discard = true;
                        break;
                    }
                }
            }

            dst = discard ? 0u : 1u;
        }
    }
}
}  // namespace Host
}  // namespace Kernels
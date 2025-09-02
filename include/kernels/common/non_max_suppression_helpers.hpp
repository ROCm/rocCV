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

namespace Kernels {
inline __device__ __host__ float ComputeArea(const short4 &box) { return box.z * box.w; }

inline __device__ __host__ float ComputeIoU(const short4 &boxA, const short4 &boxB) {
    int aInterLeft = std::max(boxA.x, boxB.x);
    int bInterTop = std::max(boxA.y, boxB.y);
    int aInterRight = std::min(boxA.x + boxA.z, boxB.x + boxB.z);
    int bInterBottom = std::min(boxA.y + boxA.w, boxB.y + boxB.w);
    int widthInter = aInterRight - aInterLeft;
    int heightInter = bInterBottom - bInterTop;
    float interArea = widthInter * heightInter;
    float iou = 0.0f;

    if (widthInter > 0.0f && heightInter > 0.0f) {
        float unionArea = ComputeArea(boxA) + ComputeArea(boxB) - interArea;
        if (unionArea > 0.0f) {
            iou = interArea / unionArea;
        }
    }

    return iou;
}
};  // namespace Kernels
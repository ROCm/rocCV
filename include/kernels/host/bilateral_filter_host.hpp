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

#include <hip/hip_runtime.h>
#include "kernels/kernel_helpers.hpp"
#include "operator_types.h"

#include "core/detail/type_traits.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/vector_utils.hpp"

namespace Kernels {
namespace Host {
                                      
template <typename T, typename SrcWrapper, typename DstWrapper>
void bilateral_filter(SrcWrapper input, DstWrapper output, int radius, float sigmaColor,
                      float sigmaSpace, int height, int width, int prevHeight, int prevWidth, 
                      float spaceCoeff, float colorCoeff) {
    using namespace roccv::detail;
    using worktype = MakeType<float,NumElements<T>>;
    for (int idz = 0; idz < output.batches(); idz++) {
        for (int idy = prevHeight; idy < height; idy += 2) {
            for (int idx = prevWidth; idx < width; idx += 2) {
                int3 coord0{idx, idy, idz};
                int3 coord1{idx + 1, idy, idz};
                int3 coord2{idx, idy + 1, idz};
                int3 coord3{idx + 1, idy + 1, idz};

                
                worktype center0 = StaticCast<worktype>(input.at(coord0.z, coord0.y, coord0.x, 0));
                worktype center1 = StaticCast<worktype>(input.at(coord1.z, coord1.y, coord1.x, 0));
                worktype center2 = StaticCast<worktype>(input.at(coord2.z, coord2.y, coord2.x, 0));
                worktype center3 = StaticCast<worktype>(input.at(coord3.z, coord3.y, coord3.x, 0));

                int sqrRadius = radius * radius;

                worktype num0 = SetAll<worktype>(0);
                worktype num1 = SetAll<worktype>(0);
                worktype num2 = SetAll<worktype>(0);
                worktype num3 = SetAll<worktype>(0);

                float4 den{0, 0, 0, 0};

                for (int c = idx - radius; c < idx + radius + 2; c++) {
                    if (c < 0 || c >= output.width()) {
                        continue;
                    }
                    for (int r = idy - radius; r < idy + radius + 2; r++) {
                        if (r < 0 || r >= output.height()) {
                            continue;
                        }
                        int t0 = abs(c - idx), t1 = abs(r - idy);
                        int t2 = abs(c - (idx + 1)), t3 = abs(r - (idy + 1));
                        float4 sqrD{t0 * t0 + t1 * t1, t2 * t2 + t1 * t1, t0 * t0 + t3 * t3, t3 * t3 + t2 * t2};

                        if (!(sqrD.x <= sqrRadius || sqrD.y <= sqrRadius ||
                              sqrD.z <= sqrRadius || sqrD.w <= sqrRadius)) {
                            continue;
                        }

                        int3 coord{c, r, idz};
                        worktype curr = StaticCast<worktype>(input.at(coord.z, coord.y, coord.x, 0));

                        if (sqrD.x <= sqrRadius) {
                            float eSpace = sqrD.x * spaceCoeff;
                            float oneNormSize = norm1(curr - center0);
                            float eColor =
                                oneNormSize * oneNormSize * colorCoeff;
                            float weight = exp(eSpace + eColor);
                            den.x += weight;
                            num0 += weight * curr;
                        }

                        if (sqrD.y <= sqrRadius) {
                            float eSpace = sqrD.y * spaceCoeff;
                            float oneNormSize = norm1(curr - center1);
                            float eColor =
                                oneNormSize * oneNormSize * colorCoeff;
                            float weight = exp(eSpace + eColor);
                            den.y += weight;
                            num1 += (weight * curr);
                        }

                        if (sqrD.z <= sqrRadius) {
                            float eSpace = sqrD.z * spaceCoeff;
                            float oneNormSize = norm1(curr - center2);
                            float eColor =
                                oneNormSize * oneNormSize * colorCoeff;
                            float weight = exp(eSpace + eColor);
                            den.z += weight;
                            num2 += (weight * curr);
                        }

                        if (sqrD.w <= sqrRadius) {
                            float eSpace = sqrD.w * spaceCoeff;
                            float oneNormSize = norm1(curr - center3);
                            float eColor =
                                oneNormSize * oneNormSize * colorCoeff;
                            float weight = exp(eSpace + eColor);
                            den.w += weight;
                            num3 += (weight * curr);
                        }
                    }
                }

                if (idx < output.width() && idy < output.height()) {
                    output.at(coord0.z, coord0.y, coord0.x, 0) = SaturateCast<T>(num0 / den.x);
                }

                if (idx + 1 < output.width() && idy < output.height()) {
                    output.at(coord1.z, coord1.y, coord1.x, 0) = SaturateCast<T>(num1 / den.y);
                }

                if (idx < output.width() && idy + 1 < output.height()) {
                    output.at(coord2.z, coord2.y, coord2.x, 0) = SaturateCast<T>(num2 / den.z);
                }

                if (idx + 1 < output.width() && idy + 1 < output.height()) {
                    output.at(coord3.z, coord3.y, coord3.x, 0) = SaturateCast<T>(num3 / den.w);
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels

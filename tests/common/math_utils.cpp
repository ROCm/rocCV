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

#include "math_utils.hpp"

namespace roccv::tests {
Point2D MatrixMultiply(const Point2D point, const std::array<float, 9>& mat) {
    // Assuming mat is a 3x3 matrix stored in row-major order
    float oX = mat[0] * point.x + mat[1] * point.y + mat[2];
    float oY = mat[3] * point.x + mat[4] * point.y + mat[5];
    float oW = mat[6] * point.x + mat[7] * point.y + mat[8];

    // Homogeneous coordinate normalization
    if (oW != 0.0f) {
        oX /= oW;
        oY /= oW;
    }

    return Point2D{oX, oY};
}
}  // namespace roccv::tests
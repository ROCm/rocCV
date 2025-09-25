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

#include <array>
#include <optional>

namespace roccv::tests {
struct Point2D {
    float x, y;
};

/**
 * @brief Transforms a given point using a 3x3 matrix multiplication.
 *
 * @param point X and y coordinates for a point in 2D space.
 * @param mat A row-major, 3x3 transformation matrix.
 * @return The transformed x and y coordinates.
 */
extern Point2D MatTransform(const Point2D point, const std::array<float, 9>& mat);

/**
 * @brief Calculates the determinant of a 3x3 matrix.
 *
 * @param m 3x3 matrix in row-major order.
 * @return The determinant of the matrix.
 */
extern float MatDet(const std::array<float, 9>& m);

/**
 * @brief Inverts a 3x3 matrix.
 *
 * @param m A 3x3 matrix in row-major order.
 * @return An optional containing the inverted 3x3 matrix in row-major order. If the matrix cannot be inverted, a
 * nullopt is returned instead.
 */
extern std::optional<std::array<float, 9>> MatInv(const std::array<float, 9>& m);

}  // namespace roccv::tests
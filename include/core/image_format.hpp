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

#include "data_type.hpp"

namespace roccv {

/**
 * @brief Acts as a container for data corresponding to how image data is laid out in memory.
 *
 */
class ImageFormat {
   public:
    explicit ImageFormat() {}
    explicit constexpr ImageFormat(eDataType dtype, int32_t num_channels)
        : m_dtype(dtype), m_num_channels(num_channels) {}

    eDataType dtype() const noexcept;
    int32_t channels() const noexcept;

   private:
    eDataType m_dtype;
    int32_t m_num_channels;
};

// Single plane with one 8-bit unsigned integer channel.
constexpr ImageFormat FMT_U8(eDataType::DATA_TYPE_U8, 1);

// Single plane with one 8-bit signed integer channel.
constexpr ImageFormat FMT_S8(eDataType::DATA_TYPE_S8, 1);

// Single plane with one 16-bit unsigned integer channel.
constexpr ImageFormat FMT_U16(eDataType::DATA_TYPE_U16, 1);

// Single plane with one 16-bit signed integer channel.
constexpr ImageFormat FMT_S16(eDataType::DATA_TYPE_S16, 1);

// Single plane with one 32-bit unsigned integer channel.
constexpr ImageFormat FMT_U32(eDataType::DATA_TYPE_U32, 1);

// Single plane with one 32-bit signed integer channel.
constexpr ImageFormat FMT_S32(eDataType::DATA_TYPE_S32, 1);

// Single plane with one 32-bit floating point channel.
constexpr ImageFormat FMT_F32(eDataType::DATA_TYPE_F32, 1);

// Single plane with one 64-bit floating point channel.
constexpr ImageFormat FMT_F64(eDataType::DATA_TYPE_F64, 1);

// Single plane with interleaved RGB 8-bit channel.
constexpr ImageFormat FMT_RGB8(eDataType::DATA_TYPE_U8, 3);

// Single plane with interleaved RGBA 8-bit channel.
constexpr ImageFormat FMT_RGBA8(eDataType::DATA_TYPE_U8, 4);

// Single plane with interleaved RGB signed 8-bit channel.
constexpr ImageFormat FMT_RGBs8(eDataType::DATA_TYPE_S8, 3);

// Single plane with interleaved RGBA signed 8-bit channel.
constexpr ImageFormat FMT_RGBAs8(eDataType::DATA_TYPE_S8, 4);

// Single plane with interleaved RGB 16-bit channel.
constexpr ImageFormat FMT_RGB16(eDataType::DATA_TYPE_U16, 3);

// Single plane with interleaved RGBA 16-bit channel.
constexpr ImageFormat FMT_RGBA16(eDataType::DATA_TYPE_U16, 4);

// Single plane with interleaved RGB signed 16-bit channel.
constexpr ImageFormat FMT_RGBs16(eDataType::DATA_TYPE_S16, 3);

// Single plane with interleaved RGBA signed 16-bit channel.
constexpr ImageFormat FMT_RGBAs16(eDataType::DATA_TYPE_S16, 4);

// Single plane with interleaved RGB 32-bit channel.
constexpr ImageFormat FMT_RGB32(eDataType::DATA_TYPE_U32, 3);

// Single plane with interleaved RGBA 32-bit channel.
constexpr ImageFormat FMT_RGBA32(eDataType::DATA_TYPE_U32, 4);

// Single plane with interleaved RGB float32 channel.
constexpr ImageFormat FMT_RGBf32(eDataType::DATA_TYPE_F32, 3);

// Single plane with interleaved RGBA float32 channel.
constexpr ImageFormat FMT_RGBAf32(eDataType::DATA_TYPE_F32, 4);

// Single plane with interleaved RGB float64 channel.
constexpr ImageFormat FMT_RGBf64(eDataType::DATA_TYPE_F64, 3);

// Single plane with interleaved RGBA float32 channel.
constexpr ImageFormat FMT_RGBAf64(eDataType::DATA_TYPE_F64, 4);

}  // namespace roccv
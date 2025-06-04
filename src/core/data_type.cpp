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

#include "core/data_type.hpp"
namespace roccv {
DataType::DataType(eDataType etype) {
    switch (etype) {
        case eDataType::DATA_TYPE_U8:
        case eDataType::DATA_TYPE_S8:
        case eDataType::DATA_TYPE_U32:
        case eDataType::DATA_TYPE_S32:
        case eDataType::DATA_TYPE_F32:
        case eDataType::DATA_TYPE_4S16:
        case eDataType::DATA_TYPE_S16:
            etype_ = etype;
            return;
        default:
            throw Exception("Invalid Tensor Data Type.", eStatusType::INVALID_VALUE);
    }

    throw Exception("Invalid Tensor Data Type.", eStatusType::INVALID_VALUE);
}

eDataType DataType::etype() const { return etype_; }

size_t DataType::size() const {
    switch (etype_) {
        case DATA_TYPE_U8:
            return sizeof(uint8_t);
        case DATA_TYPE_S8:
            return sizeof(int8_t);
        case DATA_TYPE_U32:
            return sizeof(uint32_t);
        case DATA_TYPE_S32:
            return sizeof(int32_t);
        case DATA_TYPE_F32:
            return sizeof(float);
        case DATA_TYPE_4S16:
            return sizeof(short4);
        case DATA_TYPE_S16:
            return sizeof(short);
    }

    return 0;
}

}  // namespace roccv

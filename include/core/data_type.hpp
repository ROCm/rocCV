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

#include "exception.hpp"
#include "util_enums.h"

namespace roccv {
/**
 * @brief Supported data types for use with the Tensor utilities.
 *
 */
class DataType {
   public:
    /**
     * @brief Constructs a new Data Type object
     *
     * @param[in] etype Desired data type (e.g., DATA_TYPE_U8)
     */
    explicit DataType(eDataType etype);

    /**
     * @brief Returns the eDataType enum of the DataType object.
     *
     * @return eDataType
     */
    eDataType etype() const;

    /**
     * @brief Returns the size of the data type in bytes.
     *
     * @return size_t
     */
    size_t size() const;

    bool operator==(const eDataType &rhs) const { return this->etype_ == rhs; }

    bool operator!=(const eDataType &rhs) const { return !operator==(rhs); }

    bool operator==(const DataType &rhs) const {
        return this->etype_ == rhs.etype_;
    }

    bool operator!=(const DataType &rhs) const { return !operator==(rhs); }

   private:
    eDataType etype_;
};
}  // namespace roccv
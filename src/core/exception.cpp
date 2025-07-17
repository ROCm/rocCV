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
#include "core/exception.hpp"

namespace roccv {
Exception::Exception(std::string message, eStatusType customStatus)
    : errorMessage_(message), customStatus_(customStatus) {};

Exception::Exception(eStatusType customStatus) : customStatus_(customStatus) {};

eStatusType Exception::getStatusEnum() const { return this->customStatus_; };

const char *Exception::what() const throw() { return this->errorMessage_.c_str(); }

namespace ExceptionMessage {
const char *getMessageByEnum(eStatusType status) {
    switch (status) {
        case eStatusType::INVALID_HANDLE:
            return "INVALID_HANDLE";
        case eStatusType::INVALID_POINTER:
            return "INVALID_POINTER";
        case eStatusType::INVALID_VALUE:
            return "INVALID_VALUE";
        case eStatusType::OUT_OF_BOUNDS:
            return "OUT_OF_BOUNDS";
        case eStatusType::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case eStatusType::INVALID_OPERATION:
            return "INVALID_OPERATION";
        case eStatusType::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case eStatusType::INVALID_COMBINATION:
            return "INVALID_COMBINATION";
        case eStatusType::NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        case eStatusType::NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case eStatusType::SUCCESS:
            return "SUCCESS";
            // default:
            //   return "UNKNOWN";
    }

    return "UNKNOWN";
}

const char *getMessageByEnum(eTestStatusType status) {
    switch (status) {
        case eTestStatusType::UNEXPECTED_VALUE:
            return "UNEXPECTED_VALUE";
        case eTestStatusType::TEST_FAILURE:
            return "TEST_FAILURE";
        case eTestStatusType::TEST_SUCCESS:
            return "TEST_SUCCESS";
            // default:
            //   return "UNKNOWN";
    }

    return "UNKNOWN";
}

}  // namespace ExceptionMessage
}  // namespace roccv

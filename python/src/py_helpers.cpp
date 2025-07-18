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

#include "py_helpers.hpp"

#include <stdexcept>

eDataType DLTypeToRoccvType(DLDataType dtype) {
    if (dtype.bits == 8) {
        if (dtype.code == kDLUInt) return eDataType::DATA_TYPE_U8;
        if (dtype.code == kDLInt) return eDataType::DATA_TYPE_S8;
    } else if (dtype.bits == 16) {
        if (dtype.lanes == 4) {
            return eDataType::DATA_TYPE_4S16;
        } else if (dtype.lanes == 1) {
            if (dtype.code == kDLUInt) return eDataType::DATA_TYPE_U16;
            if (dtype.code == kDLInt) return eDataType::DATA_TYPE_S16;
        }
    } else if (dtype.bits == 32) {
        if (dtype.code == kDLFloat) return eDataType::DATA_TYPE_F32;
        if (dtype.code == kDLUInt) return eDataType::DATA_TYPE_U32;
        if (dtype.code == kDLInt) return eDataType::DATA_TYPE_S32;
    } else if (dtype.bits == 64) {
        if (dtype.code == kDLFloat) return eDataType::DATA_TYPE_F64;
    }

    throw std::runtime_error("Datatype is not supported.");
}

eDeviceType DLDeviceToRoccvDevice(DLDevice device) {
    switch (device.device_type) {
        case kDLROCM:
            return eDeviceType::GPU;
        case kDLROCMHost:
        case kDLCPU:
            return eDeviceType::CPU;
        default:
            throw std::runtime_error("Tensor device type is not supported.");
    }
}

DLDevice RoccvDeviceToDLDevice(eDeviceType device) {
    // TODO: For the future, ensure that we're setting the proper device id.
    DLDevice ret;
    switch (device) {
        case eDeviceType::GPU:
            ret.device_type = kDLROCM;
            ret.device_id = 0;
            break;
        case eDeviceType::CPU:
            ret.device_type = kDLCPU;
            ret.device_id = 0;
            break;
    }

    return ret;
}

DLDataType RoccvTypeToDLType(eDataType dtype) {
    DLDataType ret;
    switch (dtype) {
        case eDataType::DATA_TYPE_F32:
            ret.bits = 32;
            ret.lanes = 1;
            ret.code = kDLFloat;
            break;
        case eDataType::DATA_TYPE_S32:
            ret.bits = 32;
            ret.lanes = 1;
            ret.code = kDLInt;
            break;
        case eDataType::DATA_TYPE_U32:
            ret.bits = 32;
            ret.lanes = 1;
            ret.code = kDLUInt;
            break;
        case eDataType::DATA_TYPE_S8:
            ret.bits = 8;
            ret.lanes = 1;
            ret.code = kDLInt;
            break;
        case eDataType::DATA_TYPE_U8:
            ret.bits = 8;
            ret.lanes = 1;
            ret.code = kDLUInt;
            break;
        case eDataType::DATA_TYPE_4S16:
            ret.bits = 16;
            ret.lanes = 4;
            ret.code = kDLInt;
            break;
        case eDataType::DATA_TYPE_S16:
            ret.bits = 16;
            ret.lanes = 1;
            ret.code = kDLInt;
            break;
        case eDataType::DATA_TYPE_U16:
            ret.bits = 16;
            ret.lanes = 1;
            ret.code = kDLUInt;
            break;
        case eDataType::DATA_TYPE_F64:
            ret.bits = 64;
            ret.lanes = 1;
            ret.code = kDLFloat;
            break;
    }

    return ret;
}

float4 GetFloat4FromPyList(py::list src) {
    if (src.size() != 4) {
        std::runtime_error("Cannot convert py::list to float4. py::list.size() != 4.");
    }
    return make_float4(src[0].cast<float>(), src[1].cast<float>(), src[2].cast<float>(), src[3].cast<float>());
}

double2 GetDouble2FromTuple(py::tuple src) {
    if (src.size() != 2) {
        std::runtime_error("Cannot convert py::tuple to double2. py::tuple.size() != 2.");
    }
    return make_double2(src[0].cast<double>(), src[1].cast<double>());
}

int2 GetInt2FromTuple(py::tuple src) {
    if (src.size() != 2) {
        std::runtime_error("Cannot convert py::tuple to int2. py::tuple.size() != 2.");
    }
    return make_int2(src[0].cast<int>(), src[1].cast<int>());
}
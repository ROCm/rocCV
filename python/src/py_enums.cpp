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

#include "py_enums.hpp"

#include <core/util_enums.h>
#include <operator_types.h>

void PyEnums::Export(py::module& m) {
    // eTensorLayout Enum Bindings
    py::enum_<eTensorLayout>(m, "eTensorLayout")
        .value("NHWC", TENSOR_LAYOUT_NHWC)
        .value("HWC", TENSOR_LAYOUT_HWC)
        .value("NC", TENSOR_LAYOUT_NC)
        .value("NW", TENSOR_LAYOUT_NW)
        .value("N", TENSOR_LAYOUT_N)
        .value("NCHW", TENSOR_LAYOUT_NCHW)
        .value("NWC", TENSOR_LAYOUT_NWC)
        .export_values();

    // eDataType Bindings
    py::enum_<eDataType>(m, "eDataType")
        .value("U8", DATA_TYPE_U8)
        .value("S8", DATA_TYPE_S8)
        .value("U32", DATA_TYPE_U32)
        .value("S32", DATA_TYPE_S32)
        .value("F32", DATA_TYPE_F32)
        .value("S16", DATA_TYPE_S16)
        .value("4S16", DATA_TYPE_4S16)
        .export_values();

    py::enum_<eDeviceType>(m, "eDeviceType")
        .value("GPU", eDeviceType::GPU)
        .value("CPU", eDeviceType::CPU)
        .export_values();

    py::enum_<eChannelType>(m, "eChannelType")
        .value("RGB", F_RGB)
        .value("BGR", BGR)
        .value("YUV", YUV)
        .value("YVU", YVU)
        .value("Grayscale", Grayscale)
        .export_values();

    // eInterpolationType Enum Bindings
    py::enum_<eInterpolationType>(m, "eInterpolationType")
        .value("NEAREST", INTERP_TYPE_NEAREST)
        .value("LINEAR", INTERP_TYPE_LINEAR)
        .value("CUBIC", INTERP_TYPE_CUBIC)
        .export_values();

    // eBorderType Enum Bindings
    py::enum_<eBorderType>(m, "eBorderType")
        .value("CONSTANT", BORDER_TYPE_CONSTANT)
        .value("REPLICATE", BORDER_TYPE_REPLICATE)
        .value("REFLECT", BORDER_TYPE_REFLECT)
        .value("WRAP", BORDER_TYPE_WRAP)
        .export_values();

    // eAxis Enum Bindings
    py::enum_<eAxis>(m, "eAxis").value("X", X).value("Y", Y).value("BOTH", BOTH).export_values();

    py::enum_<eColorSpec>(m, "eColorSpec")
        .value("BT601", BT601)
        .value("BT709", BT709)
        .value("BT2020", BT2020)
        .export_values();

    py::enum_<eThresholdType>(m, "eThresholdType")
        .value("BINARY", THRESH_BINARY)
        .value("BINARY_INV", THRESH_BINARY_INV)
        .value("TRUNC", THRESH_TRUNC)
        .value("TOZERO", THRESH_TOZERO)
        .value("TOZERO_INV", THRESH_TOZERO_INV)
        .export_values();

    py::enum_<eColorConversionCode>(m, "eColorConversionCode")
        .value("COLOR_RGB2YUV", COLOR_RGB2YUV)
        .value("COLOR_BGR2YUV", COLOR_BGR2YUV)
        .value("COLOR_YUV2RGB", COLOR_YUV2RGB)
        .value("COLOR_YUV2BGR", COLOR_YUV2BGR)
        .value("COLOR_RGB2BGR", COLOR_RGB2BGR)
        .value("COLOR_BGR2RGB", COLOR_BGR2RGB)
        .value("COLOR_RGB2GRAY", COLOR_RGB2GRAY)
        .value("COLOR_BGR2GRAY", COLOR_BGR2GRAY)
        .export_values();
        
    // eRemapType Enum Bindings
    py::enum_<eRemapType>(m, "eRemapType")
        .value("REMAP_ABSOLUTE", REMAP_ABSOLUTE)
        .value("REMAP_ABSOLUTE_NORMALIZED", REMAP_ABSOLUTE_NORMALIZED)
        .value("REMAP_RELATIVE_NORMALIZED", REMAP_RELATIVE_NORMALIZED)
        .export_values();
}

/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "operators/py_op_adv_cvt_color.hpp"

#include <op_adv_cvt_color.hpp>

namespace {

bool isSemiPlanarToInterleaved(eColorConversionCode code) {
    switch (code) {
        case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_NV12:
        case COLOR_YUV2RGB_NV21:
        case COLOR_YUV2BGR_NV21: return true;
        default: return false;
    }
}

bool isInterleavedToSemiPlanar(eColorConversionCode code) {
    switch (code) {
        case COLOR_RGB2YUV_NV12:
        case COLOR_BGR2YUV_NV12:
        case COLOR_RGB2YUV_NV21:
        case COLOR_BGR2YUV_NV21: return true;
        default: return false;
    }
}

}  // namespace

PyTensor PyOpAdvCvtColor::Execute(PyTensor& input, eColorConversionCode conversionCode, eColorSpec colorSpec,
                                  std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto inputLayout = inputTensor->layout();
    int hIdx = inputLayout.height_index();
    int cIdx = inputLayout.channels_index();

    std::vector<int64_t> outputDims;
    outputDims.reserve(inputTensor->rank());
    for (int i = 0; i < inputTensor->rank(); i++) {
        outputDims.push_back(inputTensor->shape(i));
    }

    if (isSemiPlanarToInterleaved(conversionCode)) {
        outputDims[hIdx] = (outputDims[hIdx] * 2) / 3;
        outputDims[cIdx] = 3;
    } else if (isInterleavedToSemiPlanar(conversionCode)) {
        outputDims[hIdx] = (outputDims[hIdx] * 3) / 2;
        outputDims[cIdx] = 1;
    } else {
        outputDims[cIdx] = 3;
    }

    roccv::TensorShape outputShape(inputLayout, outputDims);
    auto outputTensor = std::make_shared<roccv::Tensor>(outputShape, inputTensor->dtype(), device);

    roccv::AdvCvtColor op;
    op(hipStream, *inputTensor, *outputTensor, conversionCode, colorSpec, device);
    return PyTensor(outputTensor);
}

void PyOpAdvCvtColor::ExecuteInto(PyTensor& output, PyTensor& input, eColorConversionCode conversionCode,
                                  eColorSpec colorSpec, std::optional<std::reference_wrapper<PyStream>> stream,
                                  eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::AdvCvtColor op;
    op(hipStream, *input.getTensor(), *output.getTensor(), conversionCode, colorSpec, device);
}

void PyOpAdvCvtColor::Export(py::module& m) {
    using namespace py::literals;

    m.def("advcvtcolor", &PyOpAdvCvtColor::Execute, "src"_a, "conversion_code"_a, "color_spec"_a,
          "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Advanced Color Convert operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                conversion_code (eColorConversionCode): Conversion code specifying the formats being converted.
                color_spec (eColorSpec): Color specification selecting BT601/BT709/BT2020 conversion matrices.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("advcvtcolor_into", &PyOpAdvCvtColor::ExecuteInto, "dst"_a, "src"_a, "conversion_code"_a,
          "color_spec"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Advanced Color Convert operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                dst (rocpycv.Tensor): Output tensor for storing modified image data.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                conversion_code (eColorConversionCode): Conversion code specifying the formats being converted.
                color_spec (eColorSpec): Color specification selecting BT601/BT709/BT2020 conversion matrices.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
          )pbdoc");
}

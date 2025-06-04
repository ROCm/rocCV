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

#include "operators/py_op_custom_crop.hpp"

#include <op_custom_crop.hpp>

using namespace py::literals;

void PyOpCustomCrop::ExecuteInto(PyTensor& output, PyTensor& input, Box_t cropRect,
                                 std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::CustomCrop op;
    op(hipStream, *input.getTensor(), *output.getTensor(), cropRect, device);
}

PyTensor PyOpCustomCrop::Execute(PyTensor& input, Box_t cropRect,
                                 std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    std::vector<int64_t> outputShape(inputTensor->rank());
    outputShape[inputTensor->layout().batch_index()] = inputTensor->shape(inputTensor->layout().batch_index());
    outputShape[inputTensor->layout().height_index()] = cropRect.height;
    outputShape[inputTensor->layout().width_index()] = cropRect.width;
    outputShape[inputTensor->layout().channels_index()] = inputTensor->shape(inputTensor->layout().channels_index());

    auto outputTensor = std::make_shared<roccv::Tensor>(roccv::TensorShape(inputTensor->layout(), outputShape),
                                                        inputTensor->dtype(), inputTensor->device());

    roccv::CustomCrop op;
    op(hipStream, *inputTensor, *outputTensor, cropRect, device);
    return PyTensor(outputTensor);
}

void PyOpCustomCrop::Export(py::module& m) {
    m.def("custom_crop_into", &PyOpCustomCrop::ExecuteInto, "dst"_a, "src"_a, "crop_rect"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Custom Crop operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                crop_rect (rocpycv.Box): A Box defining how the image should be cropped.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");
    m.def("custom_crop", &PyOpCustomCrop::Execute, "src"_a, "crop_rect"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Custom Crop operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): Output tensor which image results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                crop_rect (rocpycv.Box): A Box defining how the image should be cropped.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
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

#include "operators/py_op_center_crop.hpp"

#include <op_center_crop.hpp>

#include "py_helpers.hpp"

using namespace py::literals;

PyTensor PyOpCenterCrop::Execute(PyTensor& input, py::tuple crop_size,
                                 std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    int2 work_crop_size = GetInt2FromTuple(crop_size);
    int cropWidth = work_crop_size.x;
    int cropHeight = work_crop_size.y;
    roccv::Size2D cropSize;
    cropSize.w = cropWidth;
    cropSize.h = cropHeight;

    auto inputTensor = input.getTensor();
    std::vector<int64_t> outputShape(inputTensor->rank());
    outputShape[inputTensor->layout().batch_index()] = inputTensor->shape(inputTensor->layout().batch_index());
    outputShape[inputTensor->layout().height_index()] = cropHeight;
    outputShape[inputTensor->layout().width_index()] = cropWidth;
    outputShape[inputTensor->layout().channels_index()] = inputTensor->shape(inputTensor->layout().channels_index());

    auto outputTensor = std::make_shared<roccv::Tensor>(roccv::TensorShape(inputTensor->layout(), outputShape),
                                                        inputTensor->dtype(), inputTensor->device());

    roccv::CenterCrop op;
    op(hipStream, *inputTensor, *outputTensor, cropSize, device);
    return PyTensor(outputTensor);
}

void PyOpCenterCrop::ExecuteInto(PyTensor& output, PyTensor& input, py::tuple crop_size,
                                 std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    int2 work_crop_size = GetInt2FromTuple(crop_size);
    int cropWidth = work_crop_size.x;
    int cropHeight = work_crop_size.y;
    roccv::Size2D cropSize;
    cropSize.w = cropWidth;
    cropSize.h = cropHeight;

    roccv::CenterCrop op;
    op(hipStream, *input.getTensor(), *output.getTensor(), cropSize, device);
}

void PyOpCenterCrop::Export(py::module& m) {
    m.def("center_crop", &PyOpCenterCrop::Execute, "src"_a, "crop_size"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Center Crop operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): Output tensor which image results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                crop_size (Tuple[int]): The crop rectangle width and height.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");
    
    m.def("center_crop_into", &PyOpCenterCrop::ExecuteInto, "dst"_a, "src"_a, "crop_size"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Center Crop operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): Output tensor which image results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                crop_size (Tuple[int]): The crop rectangle width and height.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
    
    
    
}
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

#include "operators/py_op_flip.hpp"

#include <op_flip.hpp>

PyTensor PyOpFlip::Execute(PyTensor& input, int32_t flipCode, std::optional<std::reference_wrapper<PyStream>> stream,
                           eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);
    roccv::Flip op;
    op(hipStream, *inputTensor, *outputTensor, flipCode, device);

    return PyTensor(outputTensor);
}

void PyOpFlip::ExecuteInto(PyTensor& output, PyTensor& input, int32_t flipCode,
                           std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::Flip op;
    op(hipStream, *input.getTensor(), *output.getTensor(), flipCode, device);
}

void PyOpFlip::Export(py::module& m) {
    using namespace py::literals;
    m.def("flip", &PyOpFlip::Execute, "src"_a, "flip_code"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU,
          R"pbdoc(
          
            Executes the Flip operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                flip_code (int): A flip code representing how images in the batch should be flipped. 
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("flip_into", &PyOpFlip::ExecuteInto, "dst"_a, "src"_a, "flip_code"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU,
          R"pbdoc(
          
            Executes the Flip operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): The destination tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                flip_code (int): A flip code representing how images in the batch should be flipped. 
                stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
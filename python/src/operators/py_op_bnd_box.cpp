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

#include "operators/py_op_bnd_box.hpp"

#include <op_bnd_box.hpp>

PyTensor PyOpBndBox::Execute(PyTensor& input, BndBoxes_t bnd_boxes,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::BndBox op;
    op(hipStream, *inputTensor, *outputTensor, bnd_boxes, device);
    return PyTensor(outputTensor);
}

void PyOpBndBox::ExecuteInto(PyTensor& output, PyTensor& input, BndBoxes_t bnd_boxes,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::BndBox op;
    op(hipStream, *input.getTensor(), *output.getTensor(), bnd_boxes, device);
}

void PyOpBndBox::Export(py::module& m) {
    using namespace py::literals;
    m.def("bndbox", &PyOpBndBox::Execute, "src"_a, "bnd_boxes"_a, 
                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the BndBox operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                bnd_boxes (rocpycv.BndBoxes): Bounding boxes to apply to input tensor.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.

            )pbdoc");
    m.def("bndbox_into", &PyOpBndBox::ExecuteInto, "dst"_a, "src"_a, "bnd_boxes"_a, 
                            "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the BndBox operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                bnd_boxes (rocpycv.BndBoxes): Bounding boxes to apply to input tensor.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
            )pbdoc");
}
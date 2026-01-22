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

#include "operators/py_op_reformat.hpp"

void PyOpReformat::ExecuteInto(PyTensor& output, PyTensor& input,
                               std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::Reformat op;
    op(hipStream, *input.getTensor(), *output.getTensor(), device);
}

PyTensor PyOpReformat::Execute(PyTensor& input, eTensorLayout outLayout,
                               std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    // TODO: Construct output tensor with the correct layout.
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::TensorShape inputShape = input.getTensor()->shape();
    roccv::TensorShape outputShape = inputShape.permute(roccv::TensorLayout(outLayout));
    auto outputTensor = std::make_shared<roccv::Tensor>(outputShape, input.getTensor()->dtype(), device);

    roccv::Reformat op;
    op(hipStream, *input.getTensor(), *outputTensor, device);
    return PyTensor(outputTensor);
}

void PyOpReformat::Export(py::module& m) {
    using namespace py::literals;

    m.def("reformat", &PyOpReformat::Execute, "input"_a, "out_layout"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
            Executes the Reformat operation and returns the result as a new tensor.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                input (rocpycv.Tensor): Input tensor to reformat.
                out_layout (rocpycv.eTensorLayout): The layout to reformat the input tensor to.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The reformatted tensor.
        )pbdoc")
        .def("reformat_into", &PyOpReformat::ExecuteInto, "output"_a, "input"_a, "stream"_a = nullptr,
             "device"_a = eDeviceType::GPU, R"pbdoc(
            Executes the Reformat operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                output (rocpycv.Tensor): Output tensor to store the result.
                input (rocpycv.Tensor): Input tensor to reformat.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
        )pbdoc");
}
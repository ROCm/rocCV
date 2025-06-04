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

#include "operators/py_op_resize.hpp"

#include <op_resize.hpp>

PyTensor PyOpResize::Execute(PyTensor& input, py::tuple shape, eInterpolationType interpolation,
                             std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();

    // Create an output tensor using the provided shape. Use the input tensor's layout and datatype.
    roccv::TensorShape outputShape(inputTensor->layout(), shape.cast<std::vector<int64_t>>());
    auto outputTensor = std::make_shared<roccv::Tensor>(outputShape, inputTensor->dtype(), device);

    roccv::Resize op;
    op(hipStream, *inputTensor, *outputTensor, interpolation, device);

    return PyTensor(outputTensor);
}

void PyOpResize::ExecuteInto(PyTensor& output, PyTensor& input, eInterpolationType interpolation,
                             std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::Resize op;
    op(hipStream, *input.getTensor(), *output.getTensor(), interpolation, device);
}

void PyOpResize::Export(py::module& m) {
    using namespace py::literals;

    m.def("resize", &PyOpResize::Execute, "src"_a, "shape"_a, "interp"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Resize operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                shape (Tuple[int]): Shape of the output tensor.
                interp (rocpycv.eInterpolationType): Interpolation type used for transform.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("resize_into", &PyOpResize::ExecuteInto, "dst"_a, "src"_a, "interp"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU,
          R"pbdoc(
          
            Executes the Resize operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): Output tensor which stores the result of the operation.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                interp (rocpycv.eInterpolationType): Interpolation type used for transform.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}

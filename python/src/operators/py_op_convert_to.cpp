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

#include "operators/py_op_convert_to.hpp"

#include <op_convert_to.hpp>

#include "py_helpers.hpp"

PyTensor PyOpConvertTo::Execute(PyTensor& input, eDataType dtype, double alpha, double beta,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();  
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), roccv::DataType(dtype), device);

    roccv::ConvertTo op;
    op(hipStream, *inputTensor, *outputTensor, alpha, beta, device);
    return PyTensor(outputTensor);                                        
}

void PyOpConvertTo::ExecuteInto(PyTensor& output, PyTensor& input, double alpha, double beta,
                                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::ConvertTo op;
    op(hipStream, *input.getTensor(), *output.getTensor(), alpha, beta, device);
}

void PyOpConvertTo::Export(py::module& m) {
    using namespace py::literals;
    m.def("convert_to",
          [](PyTensor& input, py::object dtype, double alpha, double beta,
             std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
              return PyOpConvertTo::Execute(input, DataTypeFromPyObject(dtype), alpha, beta, stream, device);
          },
          "src"_a, "dtype"_a, "alpha"_a = 1.0, "beta"_a = 0.0, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Convert To operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                dtype: Datatype of the output tensor. Either an ``rocpycv.eDataType``
                    (e.g. ``rocpycv.F32``) or a NumPy dtype/scalar type (e.g. ``np.float32``).
                alpha (double, optional): Scalar for output data. Defaults to 1.0.
                beta (double, optional): Offset for the data. Defaults to 0.0.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
        )pbdoc");
    m.def("convert_to_into", &PyOpConvertTo::ExecuteInto, "dst"_a, "src"_a, "alpha"_a = 1.0, "beta"_a = 0.0, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the Convert To operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): Output tensor which image results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                alpha (double, optional): Scalar for output data. Defaults to 1.0.
                beta (double, optional): Offset for the data. Defaults to 0.0.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
        )pbdoc");
}
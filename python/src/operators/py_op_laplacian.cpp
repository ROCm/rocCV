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

#include "operators/py_op_laplacian.hpp"

#include <op_laplacian.hpp>

#include "py_helpers.hpp"

PyTensor PyOpLaplacian::Execute(PyTensor& input, int ksize, float scale, eBorderType borderMode,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::Laplacian op;
    op(hipStream, *inputTensor, *outputTensor, ksize, scale, borderMode, device);
    return PyTensor(outputTensor);
}

void PyOpLaplacian::ExecuteInto(PyTensor& output, PyTensor& input, int ksize, float scale, eBorderType borderMode,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::Laplacian op;
    op(hipStream, *input.getTensor(), *output.getTensor(), ksize, scale, borderMode, device);
}

void PyOpLaplacian::Export(py::module& m) {
    using namespace py::literals;
    m.def("laplacian", &PyOpLaplacian::Execute, "src"_a, "ksize"_a, "scale"_a = 1.0f,
          "borderMode"_a = BORDER_TYPE_CONSTANT, py::kw_only(), "stream"_a = nullptr, "device"_a = eDeviceType::GPU,
          R"pbdoc(
        
            Executes the Laplacian operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                ksize (int): Aperture size used to compute the second-derivative filters. Must be 1 or 3.
                scale (float, optional): Scale factor for the Laplacian values. Defaults to 1 (no scale).
                borderMode (rocpycv.eBorderType, optional): The border type to identify the pixel extrapolation method. Defaults to BORDER_TYPE_CONSTANT.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("laplacian_into", &PyOpLaplacian::ExecuteInto, "dst"_a, "src"_a, "ksize"_a, "scale"_a = 1.0f,
          "borderMode"_a = BORDER_TYPE_CONSTANT, py::kw_only(), "stream"_a = nullptr, "device"_a = eDeviceType::GPU,
          R"pbdoc(
            
            Executes the Laplacian operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                ksize (int): Aperture size used to compute the second-derivative filters. Must be 1 or 3.
                scale (float, optional): Scale factor for the Laplacian values. Defaults to 1 (no scale).
                borderMode (rocpycv.eBorderType, optional): The border type to identify the pixel extrapolation method. Defaults to BORDER_TYPE_CONSTANT.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
           )pbdoc");
}
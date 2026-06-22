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

#include "operators/py_op_gaussian.hpp"

#include <op_gaussian.hpp>

#include "py_helpers.hpp"

PyTensor PyOpGaussian::Execute(PyTensor& input, const std::tuple<int, int>& kernelSize,
                               const std::tuple<double, double>& sigma, eBorderType borderMode,
                               std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    int kernelWidth = std::get<0>(kernelSize);
    int kernelHeight = std::get<1>(kernelSize);

    roccv::Gaussian op(kernelWidth, kernelHeight);
    op(hipStream, *inputTensor, *outputTensor, kernelWidth, kernelHeight, std::get<0>(sigma), std::get<1>(sigma),
       borderMode, device);
    return PyTensor(outputTensor);
}

void PyOpGaussian::ExecuteInto(PyTensor& output, PyTensor& input, const std::tuple<int, int>& kernelSize,
                               const std::tuple<double, double>& sigma, eBorderType borderMode,
                               std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    int kernelWidth = std::get<0>(kernelSize);
    int kernelHeight = std::get<1>(kernelSize);

    roccv::Gaussian op(kernelWidth, kernelHeight);
    op(hipStream, *input.getTensor(), *output.getTensor(), kernelWidth, kernelHeight, std::get<0>(sigma),
       std::get<1>(sigma), borderMode, device);
}

void PyOpGaussian::Export(py::module& m) {
    using namespace py::literals;
    m.def("gaussian", &PyOpGaussian::Execute, "src"_a, "kernelSize"_a, "sigma"_a, "borderMode"_a = BORDER_TYPE_CONSTANT,
          py::kw_only(), "stream"_a = nullptr, "device"_a = eDeviceType::GPU,
          R"pbdoc(
        
            Executes the Gaussian operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                kernelSize (Tuple[int, int]): Gaussian kernel width, height. Both kernel width and height must be odd and positive (inference from sigma not supported in Python API).
                sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions. Sigma X must be positive. If sigma Y <= 0, it will be set to sigma X.
                borderMode (rocpycv.eBorderType, optional): The border type to identify the pixel extrapolation method. Defaults to BORDER_TYPE_CONSTANT.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("gaussian_into", &PyOpGaussian::ExecuteInto, "dst"_a, "src"_a, "kernelSize"_a, "sigma"_a,
          "borderMode"_a = BORDER_TYPE_CONSTANT, py::kw_only(), "stream"_a = nullptr, "device"_a = eDeviceType::GPU,
          R"pbdoc(
            
            Executes the Gaussian operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                kernelSize (Tuple[int, int]): Gaussian kernel width, height. Both kernel width and height must be odd and positive (inference from sigma not supported in Python API).
                sigma (Tuple[double, double]): Gaussian kernel standard deviation in X,Y directions. Sigma X must be positive. If sigma Y <= 0, it will be set to sigma X.
                borderMode (rocpycv.eBorderType, optional): The border type to identify the pixel extrapolation method. Defaults to BORDER_TYPE_CONSTANT.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
           )pbdoc");
}
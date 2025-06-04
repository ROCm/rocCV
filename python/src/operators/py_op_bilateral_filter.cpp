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

#include "operators/py_op_bilateral_filter.hpp"

#include <op_bilateral_filter.hpp>

#include "py_helpers.hpp"

PyTensor PyOpBilateralFilter::Execute(PyTensor& input, int diameter, float sigmaColor, float sigmaSpace,
                            eBorderType borderMode, py::list borderValue, std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::BilateralFilter op;
    op(hipStream, *inputTensor, *outputTensor, diameter, sigmaColor, sigmaSpace, borderMode, GetFloat4FromPyList(borderValue), device);
    return PyTensor(outputTensor);
}

void PyOpBilateralFilter::ExecuteInto(PyTensor& output, PyTensor& input, int diameter, float sigmaColor, float sigmaSpace,
                            eBorderType borderMode, py::list borderValue, std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::BilateralFilter op;
    op(hipStream, *input.getTensor(), *output.getTensor(), diameter, sigmaColor, sigmaSpace, borderMode, GetFloat4FromPyList(borderValue), device);
}

void PyOpBilateralFilter::Export(py::module& m) {
    using namespace py::literals;
    m.def("bilateral_filter", &PyOpBilateralFilter::Execute, "src"_a, "diameter"_a, "sigmaColor"_a, "sigmaSpace"_a, 
                                                            "borderMode"_a, "borderValue"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
        
            Executes the Bilateral Filter operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                diameter (int): bilateral filter diameter.
                sigmaColor (float): Gaussian exponent for color difference, expected to be positive, if it isn't, will be set to 1.0
                sigmaSpace (float): Gaussian exponent for position difference expected to be positive, if it isn't, will be set to 1.0
                border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                border_value (List[float]): The color value to use when a constant border is selected.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");
    
    m.def("bilateral_filter_into", &PyOpBilateralFilter::ExecuteInto, "dst"_a, "src"_a, "diameter"_a, "sigmaColor"_a, "sigmaSpace"_a,
                                                                        "borderMode"_a, "borderValue"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            
            Executes the Bilateral Filter operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                diameter (int): bilateral filter diameter.
                sigmaColor (float): Gaussian exponent for color difference, expected to be positive, if it isn't, will be set to 1.0
                sigmaSpace (float): Gaussian exponent for position difference expected to be positive, if it isn't, will be set to 1.0
                border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                border_value (List[float]): The color value to use when a constant border is selected.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
           )pbdoc");                                   
}
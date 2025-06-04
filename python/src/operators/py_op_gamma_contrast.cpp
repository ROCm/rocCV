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

#include "operators/py_op_gamma_contrast.hpp"

#include <op_gamma_contrast.hpp>

PyTensor PyOpGammaContrast::Execute(PyTensor& input, PyTensor& gamma,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();  
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::GammaContrast op;
    op(hipStream, *inputTensor, *outputTensor, *gamma.getTensor(), device);
    return PyTensor(outputTensor);
}

void PyOpGammaContrast::ExecuteInto(PyTensor& output, PyTensor& input, PyTensor& gamma,
                                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::GammaContrast op;
    op(hipStream, *input.getTensor(), *output.getTensor(), *gamma.getTensor(), device);
}

void PyOpGammaContrast::Export(py::module& m) {
    using namespace py::literals;
    m.def("gamma_contrast", &PyOpGammaContrast::Execute, "src"_a, "gamma"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the Gamma Contrast operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                gamma (rocpycv.Tensor): One dimensional tensor containing the gamma correction values for each image in the batch.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
        )pbdoc");
    m.def("gamma_contrast_into", &PyOpGammaContrast::ExecuteInto, "dst"_a, "src"_a, "gamma"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the Gamma Contrast operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor with gamma correction applied.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                gamma (rocpycv.Tensor): One dimensional tensor containing the gamma correction values for each image in the batch.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
        )pbdoc");
}
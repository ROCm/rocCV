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

#include "operators/py_op_histogram.hpp"

#include <op_histogram.hpp>

PyTensor PyOpHistogram::Execute(PyTensor& input, std::optional<std::reference_wrapper<PyTensor>> mask,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    auto maskTensor = mask.has_value() ? std::optional<std::reference_wrapper<roccv::Tensor>>(*mask.value().get().getTensor()) : std::nullopt;

    
    auto outputTensor = std::make_shared<roccv::Tensor>(roccv::TensorShape(roccv::TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC),
                                                           {1, 256, 1}), roccv::DataType(eDataType::DATA_TYPE_S32), device);

    roccv::Histogram op;
    op(hipStream, *inputTensor, maskTensor, *outputTensor, device);
    return PyTensor(outputTensor);
}

void PyOpHistogram::ExecuteInto(PyTensor& output, PyTensor& input, std::optional<std::reference_wrapper<PyTensor>> mask,
                                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto maskTensor = mask.has_value() ? std::optional<std::reference_wrapper<roccv::Tensor>>(*mask.value().get().getTensor()) : std::nullopt;

    roccv::Histogram op;
    op(hipStream, *input.getTensor(), maskTensor, *output.getTensor(), device);
}

void PyOpHistogram::Export(py::module& m) {
    using namespace py::literals;
    m.def("histogram", &PyOpHistogram::Execute, "src"_a, "mask"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Histogram operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                mask (rocpycv.Tensor): (Optional) Mask tensor with shape equal to the input tensor shape and any value not equal 0 will be counted in histogram.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: Output tensor with width of 256 and a height equal to the batch size of input (1 if HWC input).

    )pbdoc");

    m.def("histogram_into", &PyOpHistogram::ExecuteInto, "dst"_a, "src"_a, "mask"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Histogram operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): Output tensor with width of 256 and a height equal to the batch size of input (1 if HWC input).
                src (rocpycv.Tensor): Input tensor containing one or more images.
                mask (rocpycv.Tensor): (Optional) Mask tensor with shape equal to the input tensor shape and any value not equal 0 will be counted in histogram.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None

    )pbdoc");
}                                                       
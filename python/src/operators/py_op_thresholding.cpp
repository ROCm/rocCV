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

#include "operators/py_op_thresholding.hpp"

#include <op_thresholding.hpp>

PyTensor PyOpThreshold::Execute(PyTensor& input, PyTensor& thresh, PyTensor& maxVal, uint32_t maxBatchSize, eThresholdType threshType,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::Threshold op(threshType, maxBatchSize);
    op(hipStream, *inputTensor, *outputTensor, *thresh.getTensor(), *maxVal.getTensor(), device);
    return PyTensor(outputTensor);
}

void PyOpThreshold::ExecuteInto(PyTensor& output, PyTensor& input, PyTensor& thresh, PyTensor& maxVal, uint32_t maxBatchSize, eThresholdType threshType,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::Threshold op(threshType, maxBatchSize);
    op(hipStream, *input.getTensor(), *output.getTensor(), *thresh.getTensor(), *maxVal.getTensor(), device);
}

void PyOpThreshold::Export(py::module& m) {
    using namespace py::literals;
    m.def("threshold", &PyOpThreshold::Execute, "src"_a, "thresh"_a, "maxVal"_a, "maxBatchSize"_a, "threshType"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the Thresholding operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                thresh (rocpycv.Tensor): thresh an array of size maxBatch that gives the threshold value of each image.
                maxVal (rocpycv.Tensor): maxval an array of size maxBatch that gives the maxval value of each image, using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV thresholding types.
                maxBatchSize (uint32_t): The maximum batch size.
                threshType (eThresholdType): Threshold type
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

    )pbdoc");

    m.def("threshold_into", &PyOpThreshold::ExecuteInto, "dst"_a, "src"_a, "thresh"_a, "maxVal"_a, "maxBatchSize"_a, "threshType"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Thresholding operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                thresh (rocpycv.Tensor): thresh an array of size maxBatch that gives the threshold value of each image.
                maxVal (rocpycv.Tensor): maxval an array of size maxBatch that gives the maxval value of each image, using with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV thresholding types.
                maxBatchSize (uint32_t): The maximum batch size.
                threshType (eThresholdType): Threshold type
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

    )pbdoc");
}
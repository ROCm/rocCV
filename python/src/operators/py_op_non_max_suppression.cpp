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

#include "operators/py_op_non_max_suppression.hpp"

#include <op_non_max_suppression.hpp>

PyTensor PyOpNonMaxSuppression::Execute(PyTensor& input, PyTensor& scores, float scoreThreshold, float iouThreshold,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    int64_t numBatches = inputTensor->shape(0);
    int64_t numBoxes = inputTensor->shape(1);
    auto outputTensor = std::make_shared<roccv::Tensor>(
        roccv::TensorShape(roccv::TensorLayout(eTensorLayout::TENSOR_LAYOUT_NW), {numBatches, numBoxes}),
        roccv::DataType(eDataType::DATA_TYPE_U8), device);

    roccv::NonMaximumSuppression op;
    op(hipStream, *inputTensor, *outputTensor, *scores.getTensor(), scoreThreshold, iouThreshold, device);

    return PyTensor(outputTensor);
}

void PyOpNonMaxSuppression::ExecuteInto(PyTensor& output, PyTensor& input, PyTensor& scores, float scoreThreshold,
                                        float iouThreshold, std::optional<std::reference_wrapper<PyStream>> stream,
                                        eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::NonMaximumSuppression op;
    op(hipStream, *input.getTensor(), *output.getTensor(), *scores.getTensor(), scoreThreshold, iouThreshold, device);
}

void PyOpNonMaxSuppression::Export(py::module& m) {
    using namespace pybind11::literals;
    m.def("nms", &PyOpNonMaxSuppression::Execute, "src"_a, "scores"_a,
          "score_threshold"_a = std::numeric_limits<float>::epsilon(), "iou_threshold"_a = 1.0, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Non-maximum Suppression operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): An input tensor of size [i, j, 4] containing bounding boxes with the following structure (x, y, width, height).
                scores (rocpycv.Tensor): A size [i, j] tensor containing confidence scores for each box j in batch i.
                score_threshold (float): Minimum score an input bounding box proposal needs to be kept.
                iou_threshold (float): The IoU threshold to filter overlapping boxes. Defaults to 1.0.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor of shape [i, j], containing 1 (kept) or 0 (suppressed) for each bounding box (j) per batch (i). Results will be written to this tensor.
          )pbdoc");
    m.def("nms_into", &PyOpNonMaxSuppression::ExecuteInto, "dst"_a, "src"_a, "scores"_a,
          "score_threshold"_a = std::numeric_limits<float>::epsilon(), "iou_threshold"_a = 1.0, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU,
          R"pbdoc(
          
            Executes the Non-maximum Suppression operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): The output tensor of shape [i, j], containing 1 (kept) or 0 (suppressed) for each bounding box (j) per batch (i). Results will be written to this tensor.
                src (rocpycv.Tensor): An input tensor of size [i, j, 4] containing bounding boxes with the following structure (x, y, width, height).
                scores (rocpycv.Tensor): A size [i, j] tensor containing confidence scores for each box j in batch i.
                score_threshold (float): Minimum score an input bounding box proposal needs to be kept.
                iou_threshold (float): The IoU threshold to filter overlapping boxes. Defaults to 1.0.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
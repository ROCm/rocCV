/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "operators/py_op_copy_make_border.hpp"

#include <op_copy_make_border.hpp>

#include "py_helpers.hpp"

void PyOpCopyMakeBorder::ExecuteInto(PyTensor& output, PyTensor& input, eBorderType borderMode, py::list borderValue,
                                     int top, int left, std::optional<std::reference_wrapper<PyStream>> stream,
                                     eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::CopyMakeBorder op;
    float4 borderValueFloat = GetFloat4FromPyList(borderValue);
    op(hipStream, *input.getTensor(), *output.getTensor(), top, left, borderMode, borderValueFloat, device);
}

PyTensor PyOpCopyMakeBorder::Execute(PyTensor& input, eBorderType borderMode, py::list borderValue, int top, int bottom,
                                     int left, int right, std::optional<std::reference_wrapper<PyStream>> stream,
                                     eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    // Create output tensor shape
    roccv::TensorShape inputShape = input.getTensor()->shape();
    std::vector<int64_t> shapeData(inputShape.layout().rank());
    if (inputShape.layout().batch_index() != -1) {
        shapeData[inputShape.layout().batch_index()] = inputShape[inputShape.layout().batch_index()];
    }

    shapeData[inputShape.layout().height_index()] = inputShape[inputShape.layout().height_index()] + top + bottom;
    shapeData[inputShape.layout().width_index()] = inputShape[inputShape.layout().width_index()] + left + right;
    shapeData[inputShape.layout().channels_index()] = inputShape[inputShape.layout().channels_index()];

    roccv::TensorShape outputShape(inputShape.layout(), shapeData);
    auto output = std::make_shared<roccv::Tensor>(outputShape, input.getTensor()->dtype(), device);
    float4 borderValueFloat = GetFloat4FromPyList(borderValue);

    // Run Copy Make Border on created output tensor
    roccv::CopyMakeBorder op;
    op(hipStream, *input.getTensor(), *output, top, left, borderMode, borderValueFloat, device);

    return PyTensor(output);
}

void PyOpCopyMakeBorder::Export(py::module& m) {
    using namespace py::literals;
    m.def("copymakeborder", &PyOpCopyMakeBorder::Execute, "src"_a, "border_mode"_a = eBorderType::BORDER_TYPE_CONSTANT,
          "border_value"_a = std::vector<float>(4), "top"_a, "bottom"_a, "left"_a, "right"_a, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the CopyMakeBorder operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input image tensor.
                border_mode (rocpycv.eBorderType): Border type.
                border_value (List[float]): Border values to use when using constant border type.
                top (int): Top border height in pixels.
                bottom (int): Bottom border height in pixels.
                left (int): Left border width in pixels.
                right (int): Right border width in pixels.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("copymakeborder_into", &PyOpCopyMakeBorder::ExecuteInto, "dst"_a, "src"_a,
          "border_mode"_a = eBorderType::BORDER_TYPE_CONSTANT, "border_value"_a = std::vector<float>(4), "top"_a,
          "left"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the CopyMakeBorder operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): The destination tensor.
                src (rocpycv.Tensor): Input image tensor.
                border_mode (rocpycv.eBorderType): Border type.
                border_value (List[float]): Border values to use when using constant border type.
                top (int): Top border height in pixels.
                left (int): Left border width in pixels.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
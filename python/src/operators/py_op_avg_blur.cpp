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

#include "operators/py_op_avg_blur.hpp"

#include <op_avg_blur.hpp>

#include "py_helpers.hpp"

PyTensor PyOpAvgBlur::Execute(PyTensor& input, py::tuple kernelSize, py::tuple anchor,
                              eBorderType borderMode, py::list borderValue,
                              std::optional<std::reference_wrapper<PyStream>> stream,
                              eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    // Extract kernel size from tuple
    int kernelWidth = kernelSize[0].cast<int>();
    int kernelHeight = kernelSize[1].cast<int>();

    // Extract anchor from tuple
    int kernelAnchorX = anchor[0].cast<int>();
    int kernelAnchorY = anchor[1].cast<int>();

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::AvgBlur op;
    op(hipStream, *inputTensor, *outputTensor, kernelWidth, kernelHeight,
       kernelAnchorX, kernelAnchorY, borderMode, GetFloat4FromPyList(borderValue), device);
    return PyTensor(outputTensor);
}

void PyOpAvgBlur::ExecuteInto(PyTensor& output, PyTensor& input,
                              py::tuple kernelSize, py::tuple anchor,
                              eBorderType borderMode, py::list borderValue,
                              std::optional<std::reference_wrapper<PyStream>> stream,
                              eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    // Extract kernel size from tuple
    int kernelWidth = kernelSize[0].cast<int>();
    int kernelHeight = kernelSize[1].cast<int>();

    // Extract anchor from tuple
    int kernelAnchorX = anchor[0].cast<int>();
    int kernelAnchorY = anchor[1].cast<int>();

    roccv::AvgBlur op;
    op(hipStream, *input.getTensor(), *output.getTensor(), kernelWidth, kernelHeight,
       kernelAnchorX, kernelAnchorY, borderMode, GetFloat4FromPyList(borderValue), device);
}

void PyOpAvgBlur::Export(py::module& m) {
    using namespace py::literals;
    m.def("avg_blur", &PyOpAvgBlur::Execute, "src"_a, "kernelSize"_a, "anchor"_a,
                                              "borderMode"_a, "borderValue"_a,
                                              "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Average Blur operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                kernelSize (Tuple[int, int]): Kernel size as (width, height).
                anchor (Tuple[int, int]): Kernel anchor position as (x, y).
                borderMode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                borderValue (List[float]): The color value to use when a constant border is selected.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor with blurred images.
          )pbdoc");

    m.def("avg_blur_into", &PyOpAvgBlur::ExecuteInto, "dst"_a, "src"_a, "kernelSize"_a, "anchor"_a,
                                                       "borderMode"_a, "borderValue"_a,
                                                       "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(


            Executes the Average Blur operation on the given HIP stream, writing results into a pre-allocated output tensor.

            This operation applies an average (mean) blur filter on images in a tensor.
            The filter computes the average of all pixels within a rectangular kernel
            for each pixel position in the image.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.

            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                kernelSize (Tuple[int, int]): Kernel size as (width, height).
                anchor (Tuple[int, int]): Kernel anchor position as (x, y).
                borderMode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                borderValue (List[float]): The color value to use when a constant border is selected (4 elements for RGBA).
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
           )pbdoc");
}

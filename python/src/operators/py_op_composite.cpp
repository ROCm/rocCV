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

#include "operators/py_op_composite.hpp"

#include <op_composite.hpp>

void PyOpComposite::ExecuteInto(PyTensor& dst, PyTensor& foreground, PyTensor& background, PyTensor& mask,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::Composite op;
    op(hipStream, *foreground.getTensor(), *background.getTensor(), *mask.getTensor(), *dst.getTensor(), device);
}

PyTensor PyOpComposite::Execute(PyTensor& foreground, PyTensor& background, PyTensor& mask, int out_channels,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto foregroundTensor = foreground.getTensor();
    // Out shape should match foreground shape but with the specified out_channels.
    roccv::TensorShape out_shape(foregroundTensor->layout(),
                                 {foregroundTensor->shape(foregroundTensor->layout().batch_index()),
                                  foregroundTensor->shape(foregroundTensor->layout().height_index()),
                                  foregroundTensor->shape(foregroundTensor->layout().width_index()), out_channels});
    auto output = std::make_shared<roccv::Tensor>(out_shape, foregroundTensor->dtype(), device);

    roccv::Composite op;
    op(hipStream, *foregroundTensor, *background.getTensor(), *mask.getTensor(), *output, device);

    return PyTensor(output);
}

void PyOpComposite::Export(py::module& m) {
    using namespace py::literals;
    m.def("composite", &PyOpComposite::Execute, "foreground"_a, "background"_a, "fgmask"_a, "outchannels"_a,
          "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Composite operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                foreground (rocpycv.Tensor): Input foreground image.
                background (rocpycv.Tensor): Input background image.
                fgmask (rocpycv.Tensor): Grayscale alpha mask for compositing.
                outchannels (int): Number of output channels for the output tensor. Must be 3 or 4. If 4, an alpha channel set to the max value will be added.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor with <outchannels> number of channels.
          )pbdoc");

    m.def("composite_into", &PyOpComposite::ExecuteInto, "dst"_a, "foreground"_a, "background"_a, "fgmask"_a,
          "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Composite operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): The output tensor with <outchannels> number of channels. Results will be written to this tensor.
                foreground (rocpycv.Tensor): Input foreground image.
                background (rocpycv.Tensor): Input background image.
                fgmask (rocpycv.Tensor): Grayscale alpha mask for compositing.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
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

#include "operators/py_op_normalize.hpp"

#include <op_normalize.hpp>

enum OpFlags : uint32_t { SCALE_IS_STDDEV = ROCCV_NORMALIZE_SCALE_IS_STDDEV };

PyTensor PyOpNormalize::Execute(PyTensor& input, PyTensor& base, PyTensor& scale, std::optional<uint32_t> flags,
                                float globalScale, float globalShift, float epsilon,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::Normalize op;
    op(hipStream, *inputTensor, *base.getTensor(), *scale.getTensor(), *outputTensor, globalScale, globalShift, epsilon,
       flags.value_or(0), device);
    return PyTensor(outputTensor);
}

void PyOpNormalize::ExecuteInto(PyTensor& output, PyTensor& input, PyTensor& base, PyTensor& scale,
                                std::optional<uint32_t> flags, float globalScale, float globalShift, float epsilon,
                                std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::Normalize op;
    op(hipStream, *input.getTensor(), *base.getTensor(), *scale.getTensor(), *output.getTensor(), globalScale,
       globalShift, epsilon, flags.value_or(0), device);
}

void PyOpNormalize::Export(py::module& m) {
    using namespace py::literals;

    py::enum_<OpFlags>(m, "NormalizeFlags").value("SCALE_IS_STDDEV", OpFlags::SCALE_IS_STDDEV);

    m.def("normalize", &PyOpNormalize::Execute, "src"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt,
          "globalscale"_a = 1.0f, "globalshift"_a = 0.0f, "epsilon"_a = 0.0f, "stream"_a = nullptr,
          "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Normalize operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                base (rocpycv.Tensor): Tensor for base values.
                scale (rocpycv.Tensor): Tensor for scale values.
                flags (int): Flags for the Normalize operation. Use NormalizeFlags.SCALE_IS_STDDEV to interpret the scale tensor as standard deviation instead.
                globalscale (float): Scale factor applied after the mean is subtracted and the standard deviation is divided. Defaults to 1.
                globalshift (float): The values of the final image will be shifted by this amount after scaling. Defaults to 0.
                epsilon (float): Epsilon value for numerical stability. Defaults to 0.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");

    m.def("normalize_into", &PyOpNormalize::ExecuteInto, "dst"_a, "src"_a, "base"_a, "scale"_a,
          "flags"_a = std::nullopt, "globalscale"_a = 1.0f, "globalshift"_a = 0.0f, "epsilon"_a = 0.0f,
          "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
              Executes the Normalize operation on the given HIP stream.
  
              See also:
                  Refer to the rocCV C++ API reference for more information on this operation.
          
              Args:
                  dst (rocpycv.Tensor): The output tensor which results are written to.
                  src (rocpycv.Tensor): Input tensor containing one or more images.
                  base (rocpycv.Tensor): Tensor for base values.
                  scale (rocpycv.Tensor): Tensor for scale values.
                  flags (int): Flags for the Normalize operation. Use NormalizeFlags.SCALE_IS_STDDEV to interpret the scale tensor as standard deviation instead.
                  globalscale (float): Scale factor applied after the mean is subtracted and the standard deviation is divided. Defaults to 1.
                  globalshift (float): The values of the final image will be shifted by this amount after scaling. Defaults to 0.
                  epsilon (float): Epsilon value for numerical stability. Defaults to 0.
                  stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                  device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
              
              Returns:
                  None
            )pbdoc");
}

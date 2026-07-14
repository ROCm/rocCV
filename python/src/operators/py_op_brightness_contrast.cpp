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

#include "operators/py_op_brightness_contrast.hpp"

#include <op_brightness_contrast.hpp>

PyTensor PyOpBrightnessContrast::Execute(PyTensor& input, std::optional<std::reference_wrapper<PyTensor>> brightness,
                                         std::optional<std::reference_wrapper<PyTensor>> contrast,
                                         std::optional<std::reference_wrapper<PyTensor>> brightnessShift,
                                         std::optional<std::reference_wrapper<PyTensor>> contrastCenter,
                                         std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    auto brightnessTensor =
        brightness.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*brightness.value().get().getTensor())
            : std::nullopt;
    auto contrastTensor =
        contrast.has_value() ? std::optional<std::reference_wrapper<roccv::Tensor>>(*contrast.value().get().getTensor())
                             : std::nullopt;
    auto brightnessShiftTensor =
        brightnessShift.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*brightnessShift.value().get().getTensor())
            : std::nullopt;
    auto contrastCenterTensor =
        contrastCenter.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*contrastCenter.value().get().getTensor())
            : std::nullopt;

    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);

    roccv::BrightnessContrast op;
    op(hipStream, *inputTensor, *outputTensor, brightnessTensor, contrastTensor, brightnessShiftTensor,
       contrastCenterTensor, device);
    return PyTensor(outputTensor);
}

void PyOpBrightnessContrast::ExecuteInto(PyTensor& output, PyTensor& input,
                                         std::optional<std::reference_wrapper<PyTensor>> brightness,
                                         std::optional<std::reference_wrapper<PyTensor>> contrast,
                                         std::optional<std::reference_wrapper<PyTensor>> brightnessShift,
                                         std::optional<std::reference_wrapper<PyTensor>> contrastCenter,
                                         std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto brightnessTensor =
        brightness.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*brightness.value().get().getTensor())
            : std::nullopt;
    auto contrastTensor =
        contrast.has_value() ? std::optional<std::reference_wrapper<roccv::Tensor>>(*contrast.value().get().getTensor())
                             : std::nullopt;
    auto brightnessShiftTensor =
        brightnessShift.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*brightnessShift.value().get().getTensor())
            : std::nullopt;
    auto contrastCenterTensor =
        contrastCenter.has_value()
            ? std::optional<std::reference_wrapper<roccv::Tensor>>(*contrastCenter.value().get().getTensor())
            : std::nullopt;

    roccv::BrightnessContrast op;
    op(hipStream, *input.getTensor(), *output.getTensor(), brightnessTensor, contrastTensor, brightnessShiftTensor,
       contrastCenterTensor, device);
}

void PyOpBrightnessContrast::Export(py::module& m) {
    using namespace py::literals;
    m.def("brightness_contrast", &PyOpBrightnessContrast::Execute, "src"_a, "brightness"_a = py::none(),
          "contrast"_a = py::none(), "brightness_shift"_a = py::none(), "contrast_center"_a = py::none(), py::kw_only(),
          "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the Brightness Contrast operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                brightness (rocpycv.Tensor, optional): Brightness multipliers. Can contain 1 or N values where N is the number of input images. Default: 1.0.
                contrast (rocpycv.Tensor, optional): Contrast multipliers. Can contain 1 or N values where N is the number of input images. Default: 1.0.
                brightness_shift (rocpycv.Tensor, optional): Brightness shifts. Can contain 1 or N values where N is the number of input images. Default: 0.0.
                contrast_center (rocpycv.Tensor, optional): Contrast centers. Can contain 1 or N values where N is the number of input images. Default: midpoint of input data type range.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
        )pbdoc");
    m.def("brightness_contrast_into", &PyOpBrightnessContrast::ExecuteInto, "dst"_a, "src"_a,
          "brightness"_a = py::none(), "contrast"_a = py::none(), "brightness_shift"_a = py::none(),
          "contrast_center"_a = py::none(), py::kw_only(), "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
            
            Executes the  Brightness Contrast operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                brightness (rocpycv.Tensor, optional): Brightness multipliers. Can contain 1 or N values where N is the number of input images. Default: 1.0.
                contrast (rocpycv.Tensor, optional): Contrast multipliers. Can contain 1 or N values where N is the number of input images. Default: 1.0.
                brightness_shift (rocpycv.Tensor, optional): Brightness shifts. Can contain 1 or N values where N is the number of input images. Default: 0.0.
                contrast_center (rocpycv.Tensor, optional): Contrast centers. Can contain 1 or N values where N is the number of input images. Default: midpoint of input data type range.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
        )pbdoc");
}
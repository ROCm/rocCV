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

#include "operators/py_op_cvt_color.hpp"

#include <op_cvt_color.hpp>

PyTensor PyOpCvtColor::Execute(PyTensor& input, eColorConversionCode conversionCode,
                                        std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    auto inputTensor = input.getTensor();
    int64_t batchSize = inputTensor->shape(inputTensor->layout().batch_index());
    int64_t height = inputTensor->shape(inputTensor->layout().height_index());
    int64_t width = inputTensor->shape(inputTensor->layout().width_index());
    int64_t channels = inputTensor->shape(inputTensor->layout().channels_index());
    
    if (conversionCode == COLOR_RGB2GRAY || conversionCode == COLOR_BGR2GRAY) {
        channels = 1;
    }
    roccv::TensorShape outputShape(inputTensor->layout(), {batchSize, height, width, channels});
    auto outputTensor = std::make_shared<roccv::Tensor>(outputShape, inputTensor->dtype(), device);

    roccv::CvtColor op;
    op(hipStream, *inputTensor, *outputTensor, conversionCode, device);
    return PyTensor(outputTensor);
}

void PyOpCvtColor::ExecuteInto(PyTensor& output, PyTensor& input, eColorConversionCode conversionCode,
                                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;
    roccv::CvtColor op;
    op(hipStream, *input.getTensor(), *output.getTensor(), conversionCode, device);
}

void PyOpCvtColor::Export(py::module& m) {
    using namespace py::literals;
    m.def("cvtcolor", &PyOpCvtColor::Execute, "src"_a, "conversion_code"_a, 
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
    
            Executes the Color Convert operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                conversion_code (eColorConversionCode): Conversion code specifying the formats being converted (ex. COLOR_RGB2YUV)
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");
    
    m.def("cvtcolor_into", &PyOpCvtColor::ExecuteInto, "dst"_a, "src"_a, "conversion_code"_a,  
                                                    "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(

            Executes the Color Convert operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
            
            Args:
                dst (rocpycv.Tensor): Output tensor for storing modified image data.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                conversion_code (eColorConversionCode): Conversion code specifying the formats being converted (ex. COLOR_RGB2YUV)
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

            Returns:
                None
          )pbdoc");
}
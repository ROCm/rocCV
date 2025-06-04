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

#include "operators/py_op_remap.hpp"

#include <op_remap.hpp>

#include "py_helpers.hpp"

PyTensor PyOpRemap::Execute(PyTensor& input, PyTensor& map, eInterpolationType inInterpolation, eInterpolationType mapInterpolation,
                            eRemapType mapValueType, bool alignCorners, eBorderType borderType, py::list borderValue,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {
    
    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    auto inputTensor = input.getTensor();
    auto outputTensor = std::make_shared<roccv::Tensor>(inputTensor->shape(), inputTensor->dtype(), device);
    auto mapTensor =  map.getTensor();
    
    roccv::Remap op;
    op(hipStream, *inputTensor, *outputTensor, *mapTensor, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, GetFloat4FromPyList(borderValue), device);

    return PyTensor(outputTensor);
}

void PyOpRemap::ExecuteInto(PyTensor& output, PyTensor& input, PyTensor& map, eInterpolationType inInterpolation, eInterpolationType mapInterpolation,
                            eRemapType mapValueType, bool alignCorners, eBorderType borderType, py::list borderValue,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device) {

    hipStream_t hipStream = stream.has_value() ? stream.value().get().getStream() : nullptr;

    roccv::Remap op;
    op(hipStream, *input.getTensor(), *output.getTensor(), *map.getTensor(), inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, GetFloat4FromPyList(borderValue), device);

}

void PyOpRemap::Export(py::module& m) {
    using namespace py::literals;
    m.def("remap", &PyOpRemap::Execute, "src"_a, "map"_a, "in_interpolation"_a, "map_interpolation"_a, 
                    "map_value_type"_a, "align_corners"_a, "border_type"_a, "border_value"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Remap operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                src (rocpycv.Tensor): Input tensor containing one or more images.
                map (rocpycv.Tensor): Map tensor containing absolute or relative positions for how to remap the pixels of the input tensor to the output tensor
                in_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting values from the input tensor.
                map_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting indices from the map tensor.
                map_value_type (rocpycv.eRemapType): Determines how the values in the map are interpreted.
                align_corners (bool): Set to true if corner values are aligned to center points of corner pixels and set to false if they are aligned by the corner points of the corner pixels.
                border_type (rocpycv.eBorderType): A border type to identify the pixel extrapolation method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
                border_value (List[float]): The color value to use when a constant border is selected.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                rocpycv.Tensor: The output tensor.
          )pbdoc");
    m.def("remap_into", &PyOpRemap::ExecuteInto, "dst"_a, "src"_a, "map"_a, "in_interpolation"_a, "map_interpolation"_a, 
                            "map_value_type"_a, "align_corners"_a, "border_type"_a, "border_value"_a, "stream"_a = nullptr, "device"_a = eDeviceType::GPU, R"pbdoc(
          
            Executes the Remap operation on the given HIP stream.

            See also:
                Refer to the rocCV C++ API reference for more information on this operation.
        
            Args:
                dst (rocpycv.Tensor): The output tensor which results are written to.
                src (rocpycv.Tensor): Input tensor containing one or more images.
                map (rocpycv.Tensor): Map tensor containing absolute or relative positions for how to remap the pixels of the input tensor to the output tensor
                in_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting values from the input tensor.
                map_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting indices from the map tensor.
                map_value_type (rocpycv.eRemapType): Determines how the values in the map are interpreted.
                align_corners (bool): Set to true if corner values are aligned to center points of corner pixels and set to false if they are aligned by the corner points of the corner pixels.
                border_type (rocpycv.eBorderType): A border type to identify the pixel extrapolation method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
                border_value (List[float]): The color value to use when a constant border is selected.
                stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
            Returns:
                None
          )pbdoc");
}
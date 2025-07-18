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

#include "py_structs.hpp"

#include <operator_types.h>
#include <pybind11/stl.h>

using namespace py::literals;

void PyStructs::Export(py::module& m) {
    py::class_<Box_t>(m, "Box")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int32_t, int32_t>(), "x"_a, "y"_a, "width"_a, "height"_a)
        .def_readwrite("x", &Box_t::x)
        .def_readwrite("y", &Box_t::y)
        .def_readwrite("width", &Box_t::width)
        .def_readwrite("height", &Box_t::height);
    
    py::class_<Color4_t>(m, "Color4")
        .def(py::init<>())
        .def(py::init<uint8_t, uint8_t, uint8_t, uint8_t>(), "c0"_a, "c1"_a, "c2"_a, "c3"_a)
        .def_readwrite("c0", &Color4_t::c0)
        .def_readwrite("c1", &Color4_t::c1)
        .def_readwrite("c2", &Color4_t::c2)
        .def_readwrite("c3", &Color4_t::c3);
    
    py::class_<BndBox_t>(m, "BndBox")
        .def(py::init<>())
        .def(py::init<Box_t, int32_t, Color4_t, Color4_t>(), "box"_a, "thickness"_a, "borderColor"_a, "fillColor"_a)
        .def_readwrite("box", &BndBox_t::box)
        .def_readwrite("thickness", &BndBox_t::thickness)
        .def_readwrite("borderColor", &BndBox_t::borderColor)
        .def_readwrite("fillColor", &BndBox_t::fillColor);

    py::class_<BndBoxes_t>(m, "BndBoxes")
        .def(py::init<>())
        .def(py::init<int64_t, std::vector<int32_t>, std::vector<BndBox_t>>(), "batch"_a, "numBoxes"_a, "boxes"_a)
        .def_readwrite("batch", &BndBoxes_t::batch)
        .def_readwrite("numBoxes", &BndBoxes_t::numBoxes)
        .def_readwrite("boxes", &BndBoxes_t::boxes);

    py::class_<roccv::Size2D>(m, "Size2D")
        .def(py::init<>())
        .def(py::init<int, int>(), "w"_a, "h"_a)
        .def_readwrite("w", &roccv::Size2D::w)
        .def_readwrite("h", &roccv::Size2D::h);
}
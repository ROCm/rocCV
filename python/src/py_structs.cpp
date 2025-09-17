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
    py::class_<roccv::Box_t>(m, "Box")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(), "x"_a, "y"_a, "width"_a, "height"_a)
        .def_readwrite("x", &roccv::Box_t::x)
        .def_readwrite("y", &roccv::Box_t::y)
        .def_readwrite("width", &roccv::Box_t::width)
        .def_readwrite("height", &roccv::Box_t::height);

    py::class_<roccv::ColorRGBA_t>(m, "ColorRGBA")
        .def(py::init<>())
        .def(py::init<uint8_t, uint8_t, uint8_t, uint8_t>(), "r"_a, "g"_a, "b"_a, "a"_a)
        .def_readwrite("c0", &roccv::ColorRGBA_t::r)
        .def_readwrite("c1", &roccv::ColorRGBA_t::g)
        .def_readwrite("c2", &roccv::ColorRGBA_t::b)
        .def_readwrite("c3", &roccv::ColorRGBA_t::a);

    py::class_<roccv::BndBox_t>(m, "BndBox")
        .def(py::init<>())
        .def(py::init<roccv::Box_t, int32_t, roccv::ColorRGBA_t, roccv::ColorRGBA_t>(), "box"_a, "thickness"_a,
             "borderColor"_a, "fillColor"_a)
        .def_readwrite("box", &roccv::BndBox_t::box)
        .def_readwrite("thickness", &roccv::BndBox_t::thickness)
        .def_readwrite("borderColor", &roccv::BndBox_t::borderColor)
        .def_readwrite("fillColor", &roccv::BndBox_t::fillColor);

    py::class_<roccv::BndBoxes>(m, "BndBoxes")
        .def(py::init<const std::vector<std::vector<roccv::BndBox_t>>&>(), "bndboxes"_a);

    py::class_<roccv::Size2D>(m, "Size2D")
        .def(py::init<>())
        .def(py::init<int, int>(), "w"_a, "h"_a)
        .def_readwrite("w", &roccv::Size2D::w)
        .def_readwrite("h", &roccv::Size2D::h);
}
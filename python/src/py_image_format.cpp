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

#include "py_image_format.hpp"

#include <core/data_type.hpp>
#include <core/image_format.hpp>
#include <functional>
#include <string>
#include <unordered_set>

// X-macro listing every named ImageFormat constant. Each entry is
// (python_name, FMT_ suffix). This single list drives both the attribute
// definitions and the __repr__ reverse-lookup. Formats whose name begins with a
// digit get an underscore-prefixed python name (e.g. _2F32) to remain valid
// identifiers.
#define ROCCV_FORMAT_LIST(ENTRY) \
    ENTRY("NONE", NONE)          \
    ENTRY("U8", U8)              \
    ENTRY("S8", S8)              \
    ENTRY("U16", U16)            \
    ENTRY("S16", S16)            \
    ENTRY("U32", U32)            \
    ENTRY("S32", S32)            \
    ENTRY("F32", F32)            \
    ENTRY("_2F32", 2F32)         \
    ENTRY("F64", F64)            \
    ENTRY("RGB8", RGB8)          \
    ENTRY("BGR8", BGR8)          \
    ENTRY("RGBA8", RGBA8)        \
    ENTRY("BGRA8", BGRA8)        \
    ENTRY("RGBs8", RGBs8)        \
    ENTRY("RGBAs8", RGBAs8)      \
    ENTRY("RGB16", RGB16)        \
    ENTRY("RGBA16", RGBA16)      \
    ENTRY("RGBs16", RGBs16)      \
    ENTRY("RGBAs16", RGBAs16)    \
    ENTRY("RGB32", RGB32)        \
    ENTRY("RGBA32", RGBA32)      \
    ENTRY("RGBs32", RGBs32)      \
    ENTRY("RGBAs32", RGBAs32)    \
    ENTRY("RGBf32", RGBf32)      \
    ENTRY("RGBAf32", RGBAf32)    \
    ENTRY("RGBf64", RGBf64)      \
    ENTRY("RGBAf64", RGBAf64)

namespace {

// Names of the class-level format constants (Format.RGB8, Format.U8, ...). Used
// to hide them from instance attribute lookup so chains like Format.RGB8.U8 are
// rejected (see __getattribute__ in Export).
const std::unordered_set<std::string> kConstantNames = {
#define ENTRY(pyname, suffix) pyname,
    ROCCV_FORMAT_LIST(ENTRY)
#undef ENTRY
};

// Produces a human-readable name for a format, preferring the named constant it
// matches (e.g. "rocpycv.Format.RGB8"). Falls back to a structural description
// for formats constructed directly from (dtype, channels, swizzle).
std::string ImageFormatToString(const roccv::ImageFormat& fmt) {
#define ENTRY(pyname, suffix) \
    if (fmt == roccv::FMT_##suffix) return std::string("rocpycv.Format.") + (pyname);
    ROCCV_FORMAT_LIST(ENTRY)
#undef ENTRY

    return "rocpycv.Format(dtype=" + std::to_string(static_cast<int>(fmt.dtype())) +
           ", channels=" + std::to_string(fmt.channels()) +
           ", swizzle=" + std::to_string(static_cast<int>(fmt.swizzle())) + ")";
}

size_t ImageFormatHash(const roccv::ImageFormat& fmt) {
    size_t h = std::hash<int>()(static_cast<int>(fmt.dtype()));
    h = h * 31 + std::hash<int>()(fmt.channels());
    h = h * 31 + std::hash<int>()(static_cast<int>(fmt.swizzle()));
    return h;
}

}  // namespace

void PyImageFormat::Export(py::module& m) {
    using namespace py::literals;
    using roccv::ImageFormat;

    py::class_<ImageFormat> fmt(m, "Format", "Describes how image pixel data is laid out in memory.");

    fmt.def(py::init<eDataType, int32_t, roccv::eSwizzle>(), "dtype"_a, "channels"_a,
            "swizzle"_a = roccv::eSwizzle::XYZW, "Construct a format from a data type, channel count, and swizzle.")
        .def_property_readonly("channels", &ImageFormat::channels,
                               "Read-only property that returns the number of color channels in the image.")
        .def_property_readonly("dtype", &ImageFormat::dtype,
                               "Read-only property that returns the data type of each channel.")
        .def_property_readonly("swizzle", &ImageFormat::swizzle,
                               "Read-only property that returns the channel swizzle of the format.")
        .def_property_readonly(
            "planes", [](const ImageFormat&) { return 1; },
            "Read-only property that returns the number of planes in the image (always 1; rocCV is single-plane).")
        .def("__eq__", &ImageFormat::operator==, py::is_operator())
        .def("__ne__", &ImageFormat::operator!=, py::is_operator())
        .def("__hash__", &ImageFormatHash)
        .def("__repr__", &ImageFormatToString);

    // Attach each named format constant as a class attribute, e.g. Format.RGB8.
#define ENTRY(pyname, suffix) fmt.attr(pyname) = roccv::FMT_##suffix;
    ROCCV_FORMAT_LIST(ENTRY)
#undef ENTRY

    // The constants above are themselves Format instances and live in the class
    // dict, so ordinary instance attribute lookup would re-resolve them through
    // any Format value — enabling nonsensical chains like Format.RGB8.RGB8.U8.
    // Override __getattribute__ to hide the constant names from instance lookup;
    // class-level access (Format.RGB8) goes through the metaclass and is
    // unaffected. This mirrors the instance-isolation that py::enum_ members get.
    fmt.def("__getattribute__", [](py::handle self, py::str name) -> py::object {
        if (kConstantNames.count(static_cast<std::string>(name))) {
            throw py::attribute_error("'rocpycv.Format' object has no attribute '" + static_cast<std::string>(name) +
                                      "'");
        }
        PyObject* result = PyObject_GenericGetAttr(self.ptr(), name.ptr());
        if (!result) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(result);
    });
}

#undef ROCCV_FORMAT_LIST

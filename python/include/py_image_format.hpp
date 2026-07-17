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

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Exports roccv::ImageFormat to Python as the `Format` type.
 *
 * Every FMT_* constant is exposed as a named attribute (e.g. Format.RGB8), with
 * read-only `channels`/`dtype`/`swizzle`/`planes` properties and a named
 * __repr__. Because roccv::ImageFormat is a value class (dtype + channels +
 * swizzle) rather than a packed integer, it is bound as a py::class_ instead of
 * a py::enum_.
 */
class PyImageFormat {
   public:
    static void Export(py::module& m);
};

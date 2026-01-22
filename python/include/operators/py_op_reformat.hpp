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

#include <op_reformat.hpp>

#include "py_stream.hpp"
#include "py_tensor.hpp"

namespace py = pybind11;

class PyOpReformat {
   public:
    /**
     * @brief Exports the Reformat operator to a Python module.
     *
     * @param[in] m The Python module to export the operator to.
     */
    static void Export(py::module& m);

    /**
     * @brief Defines the python wrapper for `roccv::Reformat`. Executes the Reformat operation and returns the result
     * as a new tensor.
     *
     * @param[in] input The input tensor to reformat.
     * @param[in] outLayout The layout to reformat the input tensor to.
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] device The device to run the operation on.
     * @return The result tensor.
     */
    static PyTensor Execute(PyTensor& input, eTensorLayout outLayout,
                            std::optional<std::reference_wrapper<PyStream>> stream, eDeviceType device);

    /**
     * @brief Defines the python wrapper for `roccv::Reformat`. Executes the Reformat operation and stores the result in
     * the output tensor.
     *
     * @param[out] output The output tensor to store the result.
     * @param[in] input The input tensor to reformat.
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] device The device to run the operation on.
     */
    static void ExecuteInto(PyTensor& output, PyTensor& input, std::optional<std::reference_wrapper<PyStream>> stream,
                            eDeviceType device);
};
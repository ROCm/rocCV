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

#pragma once

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyStream {
   public:
    /**
     * @brief Creates a new HIP stream and wraps it upon creation.
     *
     */
    PyStream();

    /**
     * @brief Destroys the wrapped HIP stream.
     *
     */
    ~PyStream();

    /**
     * @brief Gets the wrapped HIP stream.
     *
     * @return hipStream_t
     */
    hipStream_t getStream();

    /**
     * @brief Synchronizes with the wrapped HIP stream. This will block until all work queued on the HIP stream has been
     * completed.
     *
     */
    void synchronize();

    /**
     * @brief Exports the PyStream object to the specified python module.
     *
     * @param m The python module to export this object to.
     */
    static void Export(py::module& m);

   private:
    hipStream_t m_stream;
};
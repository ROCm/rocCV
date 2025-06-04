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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "operators/py_op_bilateral_filter.hpp"
#include "operators/py_op_bnd_box.hpp"
#include "operators/py_op_composite.hpp"
#include "operators/py_op_copy_make_border.hpp"
#include "operators/py_op_custom_crop.hpp"
#include "operators/py_op_cvt_color.hpp"
#include "operators/py_op_flip.hpp"
#include "operators/py_op_gamma_contrast.hpp"
#include "operators/py_op_histogram.hpp"
#include "operators/py_op_non_max_suppression.hpp"
#include "operators/py_op_normalize.hpp"
#include "operators/py_op_remap.hpp"
#include "operators/py_op_resize.hpp"
#include "operators/py_op_rotate.hpp"
#include "operators/py_op_thresholding.hpp"
#include "operators/py_op_warp_affine.hpp"
#include "operators/py_op_warp_perspective.hpp"
#include "py_enums.hpp"
#include "py_exception.hpp"
#include "py_stream.hpp"
#include "py_structs.hpp"
#include "py_tensor.hpp"

PYBIND11_MODULE(rocpycv, m) {
    m.doc() = R"pbdoc(
        Python API reference
        -----------------------
        This is the Python API reference for rocCV.
    )pbdoc";
    PyException::Export(m);
    PyEnums::Export(m);
    PyStructs::Export(m);
    PyStream::Export(m);
    PyTensor::Export(m);
    PyOpCustomCrop::Export(m);
    PyOpNonMaxSuppression::Export(m);
    PyOpNormalize::Export(m);
    PyOpResize::Export(m);
    PyOpRotate::Export(m);
    PyOpFlip::Export(m);
    PyOpWarpAffine::Export(m);
    PyOpWarpPerspective::Export(m);
    PyOpBilateralFilter::Export(m);
    PyOpThreshold::Export(m);
    PyOpRemap::Export(m);
    PyOpHistogram::Export(m);
    PyOpCvtColor::Export(m);
    PyOpBndBox::Export(m);
    PyOpGammaContrast::Export(m);
    PyOpComposite::Export(m);
    PyOpCopyMakeBorder::Export(m);
}
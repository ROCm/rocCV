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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "operators/py_op_bilateral_filter.hpp"
#include "operators/py_op_bnd_box.hpp"
#include "operators/py_op_center_crop.hpp"
#include "operators/py_op_composite.hpp"
#include "operators/py_op_convert_to.hpp"
#include "operators/py_op_copy_make_border.hpp"
#include "operators/py_op_custom_crop.hpp"
#include "operators/py_op_cvt_color.hpp"
#include "operators/py_op_adv_cvt_color.hpp"
#include "operators/py_op_flip.hpp"
#include "operators/py_op_gamma_contrast.hpp"
#include "operators/py_op_histogram.hpp"
#include "operators/py_op_non_max_suppression.hpp"
#include "operators/py_op_normalize.hpp"
#include "operators/py_op_reformat.hpp"
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
        rocpycv — AMD GPU-accelerated image pre/post-processing
        =======================================================

        rocpycv is the Python binding for rocCV, a HIP/ROCm image processing
        library. It exposes a NumPy-friendly :class:`Tensor` and a suite of
        operators (resize, normalize, color conversion, geometric warps, ...)
        that run on either GPU (default) or CPU.

        Quick start
        -----------
        .. code-block:: python

            import numpy as np
            import rocpycv

            # Wrap a NumPy array as a CPU Tensor (zero-copy via DLPack), then
            # copy it to the GPU (explicit H2D transfer).
            host = np.zeros((1, 480, 640, 3), np.uint8)
            src  = rocpycv.from_dlpack(host, "NHWC").copy_to(rocpycv.GPU)

            # Functional form: operators allocate and return a new Tensor.
            resized = rocpycv.resize(src, (1, 224, 224, 3), rocpycv.LINEAR)
            chw     = rocpycv.reformat(resized, "NCHW")

            # ``*_into`` form: write into a caller-allocated output, optionally
            # on a stream — useful in hot preprocessing loops.
            stream = rocpycv.Stream()
            out    = rocpycv.Tensor((1, 224, 224, 3), np.uint8, "NHWC")
            rocpycv.resize_into(out, src, rocpycv.LINEAR, stream)
            stream.synchronize()

        Tensors
        -------
        :class:`Tensor` arguments accept either rocpycv enums or familiar
        Python types:

        * ``dtype``  — ``rocpycv.F32`` or any NumPy dtype/scalar (``np.float32``).
        * ``layout`` — ``rocpycv.NHWC`` or a layout string (``"NHWC"``).

        For zero-copy interop, tensors implement the DLPack protocol — pass any
        ``__dlpack__``-supporting object (NumPy array, PyTorch tensor, ...) to
        :func:`from_dlpack`, and use :meth:`Tensor.data_ptr` to hand a raw GPU
        pointer to inference frameworks such as MIGraphX.

        Operators
        ---------
        Most operators come in two forms:

        * ``op(src, ...)``       — allocates and returns a new :class:`Tensor`.
        * ``op_into(dst, src, ...)`` — writes into a pre-allocated output,
          avoiding per-call allocation in tight loops.

        All operators accept an optional ``stream`` (a :class:`Stream` wrapping
        a ``hipStream_t``) and a ``device`` argument (defaults to GPU).
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
    PyOpCvtColor::Export(m);
    PyOpAdvCvtColor::Export(m);
    PyOpBndBox::Export(m);
    PyOpGammaContrast::Export(m);
    PyOpComposite::Export(m);
    PyOpCopyMakeBorder::Export(m);
    PyOpCenterCrop::Export(m);
    PyOpHistogram::Export(m);
    PyOpReformat::Export(m);
    PyOpConvertTo::Export(m);
}
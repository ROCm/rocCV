# ##############################################################################
# Copyright (c)  - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

import numpy as np
import pytest
import rocpycv


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.F32, rocpycv.eDataType.S32])
@pytest.mark.parametrize(
    "shape, layout",
    [
        ([2, 32, 64, 3], rocpycv.eTensorLayout.NHWC),
        ([1, 3, 16, 16], rocpycv.eTensorLayout.NCHW),
        ([8, 8, 4], rocpycv.eTensorLayout.HWC),
    ],
)
def test_tensor_basic_properties(shape, layout, dtype, device):
    tensor = rocpycv.Tensor(shape, dtype, layout, device)

    assert tensor.shape() == shape
    assert tensor.ndim() == len(shape)
    assert tensor.layout() == layout
    assert tensor.device() == device
    assert tensor.dtype() == dtype
    assert tensor.data_ptr() != 0


@pytest.mark.parametrize(
    "dtype_in, expected_dtype",
    [
        (rocpycv.eDataType.U8, rocpycv.eDataType.U8),
        (rocpycv.eDataType.F32, rocpycv.eDataType.F32),
        (np.uint8, rocpycv.eDataType.U8),
        (np.float32, rocpycv.eDataType.F32),
        (np.int32, rocpycv.eDataType.S32),
        (np.dtype("uint16"), rocpycv.eDataType.U16),
    ],
)
@pytest.mark.parametrize(
    "layout_in, expected_layout, shape",
    [
        (rocpycv.eTensorLayout.NHWC, rocpycv.eTensorLayout.NHWC, [2, 32, 64, 3]),
        ("NHWC", rocpycv.eTensorLayout.NHWC, [2, 32, 64, 3]),
        ("NCHW", rocpycv.eTensorLayout.NCHW, [1, 3, 16, 16]),
        ("HWC", rocpycv.eTensorLayout.HWC, [8, 8, 4]),
    ],
)
def test_tensor_construction_from_numpy_and_strings(dtype_in, expected_dtype, layout_in, expected_layout, shape):
    tensor = rocpycv.Tensor(shape, dtype_in, layout_in, rocpycv.eDeviceType.GPU)

    assert tensor.shape() == shape
    assert tensor.dtype() == expected_dtype
    assert tensor.layout() == expected_layout


def test_tensor_construction_invalid_dtype_raises():
    with pytest.raises(Exception):
        rocpycv.Tensor([1, 8, 8, 3], "not_a_dtype", rocpycv.eTensorLayout.NHWC, rocpycv.eDeviceType.GPU)


def test_tensor_construction_invalid_layout_raises():
    with pytest.raises(Exception):
        rocpycv.Tensor([1, 8, 8, 3], rocpycv.eDataType.U8, "ZYXW", rocpycv.eDeviceType.GPU)

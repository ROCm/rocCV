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

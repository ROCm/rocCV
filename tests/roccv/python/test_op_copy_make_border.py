# ##############################################################################
# Copyright (c)  - 2025 Advanced Micro Devices, Inc.
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

from test_helpers import generate_tensor


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.S8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.U32, rocpycv.eDataType.S32, rocpycv.eDataType.F32, rocpycv.eDataType.F64])
@pytest.mark.parametrize("border_mode", [rocpycv.eBorderType.CONSTANT, rocpycv.eBorderType.REFLECT, rocpycv.eBorderType.REPLICATE, rocpycv.eBorderType.WRAP])
@pytest.mark.parametrize("border_value", [
    [1.0, 0.0, 0.5, 0.9]
])
@pytest.mark.parametrize("top,bottom,left,right", [
    [9, 9, 9, 9],
    [1, 2, 3, 4]
])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples,height,width", [
    [1, 34, 23],
    [3, 67, 10],
    [7, 50, 8]
])
def test_op_copy_make_border(samples, height, width, channels, top, right, bottom, left, border_mode, border_value, dtype, device):
    input = generate_tensor(samples, width, height, channels, dtype, device)
    golden_output = rocpycv.Tensor([samples, height + top + bottom, width + right + left,
                                   channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    output = rocpycv.copymakeborder(input, border_mode, border_value, top, bottom, left, right, stream, device)
    rocpycv.copymakeborder_into(golden_output, input, border_mode, border_value, top, left, stream, device)
    stream.synchronize()

    assert output.shape() == golden_output.shape()

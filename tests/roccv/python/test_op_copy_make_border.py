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

from test_helpers import generate_tensor, compare_tensors


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
    output_golden = rocpycv.Tensor([samples, height + top + bottom, width + right + left,
                                   channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    output = rocpycv.copymakeborder(input, border_mode, border_value, top, bottom, left, right, stream, device)
    rocpycv.copymakeborder_into(output_golden, input, border_mode, border_value, top, left, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("top,bottom,left,right", [
    [-1, 0, -2, 0],
    [0, -1, 0, 0],
    [0, 0, 0, -1],
])
def test_op_copy_make_border_negative_border(top, bottom, left, right, device):
    input = generate_tensor(1, 16, 16, 3, rocpycv.eDataType.U8, device)
    with pytest.raises(rocpycv.Exception):
        rocpycv.copymakeborder(input, rocpycv.eBorderType.CONSTANT, [0, 0, 0, 0], top, bottom, left, right, None,
                               device)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("out_height,out_width,top,left", [
    [15, 14, -1, -2],  # Negative top/left
    [15, 16, 0, 0],    # Output too small to hold the input
    [20, 20, 5, 0],    # Output too small to hold the input at the given offset
])
def test_op_copy_make_border_into_invalid_border(out_height, out_width, top, left, device):
    input = generate_tensor(1, 16, 16, 3, rocpycv.eDataType.U8, device)
    output = rocpycv.Tensor([1, out_height, out_width, 3], rocpycv.eTensorLayout.NHWC, rocpycv.eDataType.U8, device)
    with pytest.raises(rocpycv.Exception):
        rocpycv.copymakeborder_into(output, input, rocpycv.eBorderType.CONSTANT, [0, 0, 0, 0], top, left, None, device)

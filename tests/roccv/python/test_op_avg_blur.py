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

from test_helpers import compare_tensors, generate_tensor


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.S32, rocpycv.eDataType.F32])
@pytest.mark.parametrize("border_mode", [rocpycv.eBorderType.CONSTANT])
@pytest.mark.parametrize("border_val", [[0, 0, 0, 0]])
@pytest.mark.parametrize("kernel_size,anchor", [
    ((3, 3), (1, 1)),   
    ((7, 7), (3, 3)),   
    ((9, 9), (4, 4))
])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples,height,width", [
    (1, 56, 64),
    (3, 14, 40),
    (5, 45, 105)
])
def test_op_avg_blur(samples, height, width, channels, border_mode, border_val, kernel_size, anchor, dtype, device):
    input = generate_tensor(samples, width, height, channels, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    rocpycv.avg_blur_into(output_golden, input, kernel_size, anchor,
                          border_mode, border_val, stream, device)
    output = rocpycv.avg_blur(input, kernel_size, anchor,
                              border_mode, border_val, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

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
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.S8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.U32, rocpycv.eDataType.S32, rocpycv.eDataType.F32])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples,height,width", [
    (1, 45, 23),
    (3, 67, 85),
    (7, 25, 95)
])
def test_op_normalize(samples, height, width, channels, device, dtype):
    input = generate_tensor(samples, width, height, channels, dtype, device)
    base = generate_tensor(1, 1, 1, channels, rocpycv.eDataType.F32, device)
    scale = generate_tensor(1, 1, 1, channels, rocpycv.eDataType.F32, device)
    output_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    rocpycv.normalize_into(output_golden, input, base, scale, None, 1.0, 0.0, 0.0, stream, device)
    output = rocpycv.normalize(input, base, scale, None, 1.0, 0.0, 0.0, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

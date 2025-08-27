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


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.CPU, rocpycv.eDeviceType.GPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.S32, rocpycv.eDataType.F32])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("flip_code", [-1, 0, 1])
@pytest.mark.parametrize("samples,width,height", [
    (1, 65, 30),
    (3, 45, 20),
    (5, 23, 106),
    (20, 104, 234),
])
def test_op_flip(samples, width, height, channels, dtype, flip_code, device):
    input_tensor = generate_tensor(samples, width, height, channels, dtype, device)
    stream = rocpycv.Stream()
    output_tensor_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)
    rocpycv.flip_into(output_tensor_golden, input_tensor, flip_code, stream, device)
    output_tensor = rocpycv.flip(input_tensor, flip_code, stream, device)
    stream.synchronize()

    assert output_tensor_golden.shape() == output_tensor.shape()

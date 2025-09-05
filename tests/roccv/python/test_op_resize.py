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
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.F32])
@pytest.mark.parametrize("interp", [rocpycv.eInterpolationType.NEAREST, rocpycv.eInterpolationType.LINEAR])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples", [1, 3, 7])
@pytest.mark.parametrize("in_shape", [[50, 100], [38, 12], [86, 34]])
@pytest.mark.parametrize("out_shape", [[10, 67], [50, 100], [50, 25]])
def test_op_resize(out_shape, in_shape, samples, channels, interp, dtype, device):
    # Input/Output shapes are passed in as format [width, height]
    input = generate_tensor(samples, in_shape[0], in_shape[1], channels, dtype, device)
    output_shape = (samples, out_shape[1], out_shape[0], channels)
    output_golden = rocpycv.Tensor(output_shape, rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    rocpycv.resize_into(output_golden, input, interp, stream, device)
    output = rocpycv.resize(input, output_shape, interp, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

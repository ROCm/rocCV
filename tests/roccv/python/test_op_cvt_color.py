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
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8])
@pytest.mark.parametrize("code", [
    rocpycv.eColorConversionCode.COLOR_RGB2GRAY,
    rocpycv.eColorConversionCode.COLOR_BGR2GRAY,
    rocpycv.eColorConversionCode.COLOR_BGR2RGB,
    rocpycv.eColorConversionCode.COLOR_BGR2YUV,
    rocpycv.eColorConversionCode.COLOR_RGB2BGR,
    rocpycv.eColorConversionCode.COLOR_RGB2YUV,
    rocpycv.eColorConversionCode.COLOR_YUV2BGR,
    rocpycv.eColorConversionCode.COLOR_YUV2RGB,
])
@pytest.mark.parametrize("samples,width,height", [
    [1, 50, 100],
    [3, 150, 50],
    [7, 15, 23]
])
def test_op_cvtcolor(samples, height, width, code, dtype, device):
    in_channels = 3
    out_channels = 3
    if code == rocpycv.eColorConversionCode.COLOR_BGR2GRAY or code == rocpycv.eColorConversionCode.COLOR_RGB2GRAY:
        out_channels = 1

    input = generate_tensor(samples, width, height, in_channels, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, out_channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    stream = rocpycv.Stream()
    output = rocpycv.cvtcolor(input, code, stream, device)
    rocpycv.cvtcolor_into(output_golden, input, code, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

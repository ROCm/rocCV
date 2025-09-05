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
import math

from test_helpers import generate_tensor, compare_tensors


def calc_center_shift(center_x, center_y, angle) -> tuple[float, float]:
    x_shift = (1 - math.cos(angle * math.pi / 180)) * center_x - math.sin(angle * math.pi / 180) * center_y
    y_shift = math.sin(angle * math.pi / 180) * center_x + (1 - math.cos(angle * math.pi / 180)) * center_y
    return (x_shift, y_shift)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("interp", [rocpycv.eInterpolationType.NEAREST, rocpycv.eInterpolationType.LINEAR])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.S8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.U32, rocpycv.eDataType.S32, rocpycv.eDataType.F32, rocpycv.eDataType.F64])
@pytest.mark.parametrize("angle", [-10, 90, 0, 360])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples,width,height", [
    [1, 50, 100],
    [3, 150, 50],
    [7, 15, 23]
])
def test_op_rotate(samples, width, height, channels, angle, dtype, interp, device):
    input = generate_tensor(samples, width, height, channels, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)

    center_x = math.floor((width + 1) / 2)
    center_y = math.floor((height + 1) / 2)
    shift = calc_center_shift(center_x, center_y, angle)

    stream = rocpycv.Stream()
    rocpycv.rotate_into(output_golden, input, angle, shift, interp, stream, device)
    output = rocpycv.rotate(input, angle, shift, interp, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

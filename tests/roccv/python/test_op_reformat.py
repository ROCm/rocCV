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

import rocpycv
import pytest

from test_helpers import compare_tensors, generate_tensor_generic

def create_tensor_shape(layout: rocpycv.eTensorLayout, samples: int, channels: int, height: int, width: int) -> list[int]:
    if layout == rocpycv.eTensorLayout.NHWC:
        return [samples, height, width, channels]
    elif layout == rocpycv.eTensorLayout.NCHW:
        return [samples, channels, height, width]
    elif layout == rocpycv.eTensorLayout.HWC:
        return [height, width, channels]
    elif layout == rocpycv.eTensorLayout.CHW:
        return [channels, height, width]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.S8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.U32, rocpycv.eDataType.S32, rocpycv.eDataType.F32])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("inLayout,outLayout", [
    (rocpycv.eTensorLayout.NHWC, rocpycv.eTensorLayout.NCHW),
    (rocpycv.eTensorLayout.NCHW, rocpycv.eTensorLayout.NHWC),
    (rocpycv.eTensorLayout.HWC, rocpycv.eTensorLayout.NHWC),
    (rocpycv.eTensorLayout.CHW, rocpycv.eTensorLayout.NCHW),
])
@pytest.mark.parametrize("samples,height,width", [
    (1, 45, 23)
])
def test_op_reformat(samples, height, width, channels, inLayout, outLayout, device, dtype):
    input_shape = create_tensor_shape(inLayout, samples, channels, height, width)
    output_shape = create_tensor_shape(outLayout, samples, channels, height, width)
    input_tensor = generate_tensor_generic(input_shape, inLayout, dtype, device)
    output_golden = rocpycv.Tensor(output_shape, dtype, outLayout, device)

    stream = rocpycv.Stream()
    rocpycv.reformat_into(input_tensor, output_golden, stream, device)
    output_tensor = rocpycv.reformat(input_tensor, outLayout, stream, device)
    stream.synchronize()

    compare_tensors(output_tensor, output_golden)
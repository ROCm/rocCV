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
import numpy as np

from test_helpers import generate_tensor, compare_tensors

@pytest.mark.parametrize("device", [rocpycv.eDeviceType.CPU, rocpycv.eDeviceType.GPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8])
@pytest.mark.parametrize("border_mode", [rocpycv.eBorderType.CONSTANT, rocpycv.eBorderType.REPLICATE, rocpycv.eBorderType.REFLECT, rocpycv.eBorderType.WRAP])
@pytest.mark.parametrize("border_val", [[255, 0, 255, 0]])
@pytest.mark.parametrize("interp", [rocpycv.eInterpolationType.NEAREST, rocpycv.eInterpolationType.LINEAR])
@pytest.mark.parametrize("map_interp", [rocpycv.eInterpolationType.NEAREST, rocpycv.eInterpolationType.LINEAR])
@pytest.mark.parametrize("map_type", [rocpycv.REMAP_ABSOLUTE])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("samples, width, height", [
    (1, 720, 480),
    (3, 200, 200),
    (7, 100, 50)
])

def test_op_remap(samples, width, height, channels, dtype, map_interp, interp, map_type, align_corners, border_mode, border_val, device):
    input_tensor = generate_tensor(samples, width, height, channels, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)


    map_list = []
    halfWidth = int(width / 2)
    for y in range(height):
        row_list = []
        x = 0
        while x < halfWidth:
            row_list.append([halfWidth - x, y])
            x += 1
        while x < width:
            row_list.append([x, y])
            x += 1
        map_list.append(row_list)

    map_np_array = np.expand_dims(np.array(map_list, np.float32), axis=0)
    remap_tensor = rocpycv.from_dlpack(map_np_array, rocpycv.NHWC).copy_to(device)

    stream = rocpycv.Stream()
    output = rocpycv.remap(input_tensor, remap_tensor, interp, map_interp, map_type, align_corners, border_mode, border_val, stream, device)
    rocpycv.remap_into(output_golden, input_tensor, remap_tensor, interp, map_interp, map_type, align_corners, border_mode, border_val, stream, device)
    stream.synchronize()

    assert output.shape() == output_golden.shape()
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
@pytest.mark.parametrize("map_type", [rocpycv.REMAP_ABSOLUTE, rocpycv.REMAP_ABSOLUTE_NORMALIZED, rocpycv.REMAP_RELATIVE_NORMALIZED])
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


    if (map_type == rocpycv.REMAP_ABSOLUTE):
        halfWidth = width // 2

        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        x_map = x_coords.copy()
        x_map[:, :halfWidth] = halfWidth - x_coords[:, :halfWidth]

        map_single = np.stack([x_map, y_coords], axis=-1)

        map_list = np.tile(map_single[np.newaxis, :, :, :], (samples, 1, 1, 1))
    elif (map_type == rocpycv.REMAP_ABSOLUTE_NORMALIZED):

        y_coords, x_coords = np.meshgrid(
        np.arange(height), 
        np.arange(width), 
        indexing='ij'
        )

        normX = (2.0 * x_coords / (width - 1)) - 1.0
        normY = (2.0 * y_coords / (height - 1)) - 1.0

        srcX = -normX
        srcY = -normY

        map_single = np.stack([srcX, srcY], axis=-1)

        map_list = np.tile(map_single[np.newaxis, :, :, :], (samples, 1, 1, 1)).astype(np.float32)
    elif (map_type == rocpycv.REMAP_RELATIVE_NORMALIZED):
        y_coords, x_coords = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )

        normX = (2.0 * x_coords / (width - 1)) - 1.0
        normY = (2.0 * y_coords / (height - 1)) - 1.0

        offsetX = -normX
        offsetY = -normY

        map_single = np.stack([offsetX, offsetY], axis=-1)

        map_list = np.tile(map_single[np.newaxis, :, :, :], (samples, 1, 1, 1)).astype(np.float32)

    map_np_array = np.array(map_list, np.float32)
    remap_tensor = rocpycv.from_dlpack(map_np_array, rocpycv.NHWC).copy_to(device)

    stream = rocpycv.Stream()
    output = rocpycv.remap(input_tensor, remap_tensor, interp, map_interp, map_type, align_corners, border_mode, border_val, stream, device)
    rocpycv.remap_into(output_golden, input_tensor, remap_tensor, interp, map_interp, map_type, align_corners, border_mode, border_val, stream, device)
    stream.synchronize()

    assert output.shape() == output_golden.shape()
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

from test_helpers import load_image, compare_image

@pytest.mark.parametrize("input_path, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, borderValue, device, expected_path, err", [
    ("test_input.bmp", rocpycv.NEAREST, rocpycv.NEAREST, rocpycv.REMAP_ABSOLUTE, False, rocpycv.CONSTANT, [255, 0, 255, 0], rocpycv.GPU, "expected_remap.bmp", 1.0),
    ("test_input.bmp", rocpycv.NEAREST, rocpycv.NEAREST, rocpycv.REMAP_ABSOLUTE, False, rocpycv.CONSTANT, [255, 0, 255, 0], rocpycv.CPU, "expected_remap.bmp", 1.0)
])

def test_op_remap(pytestconfig, input_path, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, borderValue, device, expected_path, err):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)

    width = 720
    height = 480

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

    output_tensor = rocpycv.remap(input_tensor, remap_tensor, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, borderValue, stream, device)
    stream.synchronize()

    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", err)

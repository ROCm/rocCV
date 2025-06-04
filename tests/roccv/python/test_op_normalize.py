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


@pytest.mark.parametrize("input_path, base, scale, flags, globalscale, globalshift, epsilon, device, expected_path, err", [
    ("test_input.bmp", [121.816, 117.935, 98.395], [82.195, 62.885, 61.023], rocpycv.NormalizeFlags.SCALE_IS_STDDEV,
     85.0, 180.0, 0.0, rocpycv.GPU, "expected_normalize.bmp", 0.0),
    ("test_input.bmp", [121.816, 117.935, 98.395], [82.195, 62.885, 61.023], rocpycv.NormalizeFlags.SCALE_IS_STDDEV,
     85.0, 180.0, 0.0, rocpycv.CPU, "expected_normalize.bmp", 0.0)
])
def test_op_normalize(pytestconfig, input_path, base, scale, flags, globalscale, globalshift, epsilon, device, expected_path, err):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)
    stream = rocpycv.Stream()

    stream.synchronize()
    base_tensor = rocpycv.from_dlpack(np.array([[[base]]], np.float32), rocpycv.NHWC).copy_to(device)
    scale_tensor = rocpycv.from_dlpack(np.array([[[scale]]], np.float32), rocpycv.NHWC).copy_to(device)
    output_tensor = rocpycv.normalize(input_tensor, base_tensor, scale_tensor, flags,
                                      globalscale, globalshift, epsilon, stream, device)
    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", err)

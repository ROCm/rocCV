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

from test_helpers import load_image, compare_image

@pytest.mark.parametrize("input_path, shape, interp, device, expected_path", [
    ("test_input.bmp", (1, 1920, 2160, 3), rocpycv.NEAREST, rocpycv.GPU, "expected_resize.bmp"),
    ("test_input.bmp", (1, 1920, 2160, 3), rocpycv.NEAREST, rocpycv.CPU, "expected_resize.bmp")
])
def test_op_resize(pytestconfig, input_path, shape, interp, device, expected_path):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)
    stream = rocpycv.Stream()
    output_tensor = rocpycv.resize(input_tensor, shape, interp, stream, device)
    stream.synchronize()
    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}")
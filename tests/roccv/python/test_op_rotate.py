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

from test_helpers import load_image, compare_image


def calc_center_shift(center_x, center_y, angle) -> tuple[float, float]:
    x_shift = (1 - math.cos(angle * math.pi / 180)) * center_x - math.sin(angle * math.pi / 180) * center_y
    y_shift = math.sin(angle * math.pi / 180) * center_x + (1 - math.cos(angle * math.pi / 180)) * center_y
    return (x_shift, y_shift)


@pytest.mark.parametrize("input_path, angle, interp, device, expected_path", [
    ("test_input.bmp", 315.0,
     rocpycv.NEAREST, rocpycv.GPU, "rotate/expected_rotate_nearest.bmp"),
    ("test_input.bmp", 315.0,
     rocpycv.NEAREST, rocpycv.CPU, "rotate/expected_rotate_nearest.bmp")
])
def test_op_rotate(pytestconfig, input_path, angle, interp, device, expected_path):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)
    stream = rocpycv.Stream()

    center_x = math.floor((input_tensor.shape()[2] + 1) / 2)
    center_y = math.floor((input_tensor.shape()[1] + 1) / 2)
    shift = calc_center_shift(center_x, center_y, angle)
    output_tensor = rocpycv.rotate(input_tensor, angle, shift, interp, stream, device)

    stream.synchronize()
    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", 0.0)

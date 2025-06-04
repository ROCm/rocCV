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

@pytest.mark.parametrize("input_path, conversion_code, device, expected_path, err", [
    ("test_input.bmp", rocpycv.COLOR_BGR2YUV, rocpycv.GPU, "expected_color_cvt_yuv.bmp", 1.0),
    ("expected_color_cvt_yuv.bmp", rocpycv.COLOR_YUV2BGR, rocpycv.GPU, "expected_color_cvt_bgr.bmp", 1.0),
    ("test_input.bmp", rocpycv.COLOR_BGR2RGB, rocpycv.GPU, "expected_color_cvt_rgb.bmp", 1.0),
    ("test_input.bmp", rocpycv.COLOR_BGR2GRAY, rocpycv.GPU, "expected_color_cvt_grayscale.bmp", 1.0),
    ("test_input.bmp", rocpycv.COLOR_BGR2YUV, rocpycv.CPU, "expected_color_cvt_yuv.bmp", 1.0),
    ("expected_color_cvt_yuv.bmp", rocpycv.COLOR_YUV2BGR, rocpycv.CPU, "expected_color_cvt_bgr.bmp", 1.0),
    ("test_input.bmp", rocpycv.COLOR_BGR2RGB, rocpycv.CPU, "expected_color_cvt_rgb.bmp", 1.0),
    ("test_input.bmp", rocpycv.COLOR_BGR2GRAY, rocpycv.CPU, "expected_color_cvt_grayscale.bmp", 1.0)
])

def test_op_ColorCvt(pytestconfig, input_path, conversion_code, device, expected_path, err):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)

    stream = rocpycv.Stream()

    output_tensor = rocpycv.cvtcolor(input_tensor, conversion_code, stream, device)
    stream.synchronize()

    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", err)
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

@pytest.mark.parametrize("input_path, thresh, mvdata, maxBatchSize, threshType, device, expected_path, err", [
    ("test_input.bmp", [100], [255], 1, rocpycv.BINARY, rocpycv.GPU, "expected_thresholding_binary.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.BINARY_INV, rocpycv.GPU, "expected_thresholding_binary_inv.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TOZERO_INV, rocpycv.GPU, "expected_thresholding_to_zero_inv.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TOZERO, rocpycv.GPU, "expected_thresholding_to_zero.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TRUNC, rocpycv.GPU, "expected_thresholding_trunc.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.BINARY, rocpycv.CPU, "expected_thresholding_binary.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.BINARY_INV, rocpycv.CPU, "expected_thresholding_binary_inv.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TOZERO_INV, rocpycv.CPU, "expected_thresholding_to_zero_inv.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TOZERO, rocpycv.CPU, "expected_thresholding_to_zero.bmp", 1.0),
    ("test_input.bmp", [100], [255], 1, rocpycv.TRUNC, rocpycv.CPU, "expected_thresholding_trunc.bmp", 1.0)
])

def test_op_thresholding(pytestconfig, input_path, thresh, mvdata, maxBatchSize, threshType, device, expected_path, err):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)
    thresh_tensor = rocpycv.from_dlpack(np.array(thresh, np.float64), rocpycv.N).copy_to(device)
    maxval_tensor = rocpycv.from_dlpack(np.array(mvdata, np.float64), rocpycv.N).copy_to(device)

    stream = rocpycv.Stream()

    output_tensor = rocpycv.threshold(input_tensor, thresh_tensor, maxval_tensor, maxBatchSize, threshType, stream, device)
    stream.synchronize()

    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", err)
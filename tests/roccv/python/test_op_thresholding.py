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

@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8, rocpycv.eDataType.U16, rocpycv.eDataType.S16, rocpycv.eDataType.F32, rocpycv.eDataType.F64])
@pytest.mark.parametrize("threshType", [rocpycv.BINARY, rocpycv.BINARY_INV, rocpycv.TOZERO_INV, rocpycv.TOZERO, rocpycv.TRUNC])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("thresh", [100])
@pytest.mark.parametrize("mvdata", [255])
@pytest.mark.parametrize("samples,height,width", [
    [1, 100, 200],
    [3, 300, 200],
    [7, 300, 468]
])

def test_op_thresholding(samples, height, width, channels, dtype, thresh, mvdata, threshType, device):
    input_tensor = generate_tensor(samples, width, height, channels, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)
    
    thresh_array = np.full(samples, thresh, np.float64)
    maxval_array = np.full(samples, mvdata, np.float64)

    thresh_tensor = rocpycv.from_dlpack(np.array(thresh_array, np.float64), rocpycv.N).copy_to(device)
    maxval_tensor = rocpycv.from_dlpack(np.array(maxval_array, np.float64), rocpycv.N).copy_to(device)

    stream = rocpycv.Stream()
    output = rocpycv.threshold(input_tensor, thresh_tensor, maxval_tensor, samples, threshType, stream, device)
    rocpycv.threshold_into(output_golden, input_tensor, thresh_tensor, maxval_tensor, samples, threshType, stream, device)
    stream.synchronize()

    assert output.shape() == output_golden.shape()
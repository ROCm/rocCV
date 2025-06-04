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

from test_helpers import compare_list


@pytest.mark.parametrize("boxes, scores, score_threshold, iou_threshold, device, expected, err", [
    ([[[100, 100, 200, 200], [110, 110, 210, 210], [220, 220, 320, 320], [50,  50,  150, 150]]],
     [[0.9, 0.8, 0.85, 0.7]], 0.01, 0.5, rocpycv.GPU, "expected_nms.bin", 0.0),
    ([[[100, 100, 200, 200], [110, 110, 210, 210], [220, 220, 320, 320], [50,  50,  150, 150]]],
     [[0.9, 0.8, 0.85, 0.7]], 0.01, 0.5, rocpycv.CPU, "expected_nms.bin", 0.0)
])
def test_op_non_max_suppression(pytestconfig, boxes, scores, score_threshold, iou_threshold, device, expected, err):
    # Use numpy/dlpack to wrap provided list data into tensors for use with the NMS operator
    boxes_tensor = rocpycv.from_dlpack(np.array(boxes, np.short), rocpycv.NWC).copy_to(device)
    scores_tensor = rocpycv.from_dlpack(np.array(scores, np.float32), rocpycv.NW).copy_to(device)

    stream = rocpycv.Stream()
    output_tensor = rocpycv.nms(boxes_tensor, scores_tensor, score_threshold, iou_threshold, stream, device)
    stream.synchronize()

    # Load binary data from expected binary file into a list
    with open(f"{pytestconfig.getoption('data_dir')}/{expected}", "rb") as f:
        expected_list = list(f.read())

    compare_list(output_tensor, expected_list, err)

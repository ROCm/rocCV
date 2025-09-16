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

from test_helpers import compare_tensors, generate_tensor_generic


def generate_boxes(samples: int, num_boxes: int, device: rocpycv.eDeviceType) -> rocpycv.Tensor:
    np_array = np.random.randint(10, 300, size=(samples, num_boxes, 4), dtype=np.short)
    return rocpycv.from_dlpack(np_array, rocpycv.eTensorLayout.NWC).copy_to(device)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("samples,num_boxes", [
                         (1, 4),
                         (3, 64),
                         (20, 837),
                         (1000, 34)
                         ]
                         )
def test_op_non_max_suppression(samples, num_boxes, device):
    boxes = generate_boxes(samples, num_boxes, device)
    scores = generate_tensor_generic([samples, num_boxes], rocpycv.eTensorLayout.NW, rocpycv.eDataType.F32, device)
    output_golden = rocpycv.Tensor([samples, num_boxes], rocpycv.eTensorLayout.NW, rocpycv.eDataType.U8, device)

    stream = rocpycv.Stream()
    # Hardcoding the score and IoU threshold here. The only thing we care about is the resulting size of the
    # output tensors to compare against.
    output = rocpycv.nms(boxes, scores, 0.5, 0.75, stream, device)
    rocpycv.nms_into(output_golden, boxes, scores, 0.5, 0.75, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

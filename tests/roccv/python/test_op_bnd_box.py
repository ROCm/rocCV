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
import random

from test_helpers import generate_tensor, compare_tensors


def generate_boxes(samples: int, height: int, width: int) -> rocpycv.BndBoxes:
    num_boxes = [random.randint(1, 3) for _ in range(samples)]
    boxes = []
    for sample in range(samples):
        box_list = []
        for _ in range(num_boxes[sample]):
            rand_color = rocpycv.ColorRGBA(random.randint(0, 255), random.randint(0, 255),
                                           random.randint(0, 255), random.randint(0, 255))
            rand_box = rocpycv.Box(0, 0, random.randint(0, width - 5), random.randint(0, height - 5))
            box_list.append(rocpycv.BndBox(rand_box, random.randint(0, 2), rand_color, rand_color))
        boxes.append(box_list)
    return rocpycv.BndBoxes(boxes)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("samples,height,width", [
    (1, 50, 100),
    (3, 150, 50),
    (7, 15, 23)
])
def test_op_remap(samples, height, width, channels, device):
    input = generate_tensor(samples, width, height, channels, rocpycv.eDataType.U8, device)
    boxes = generate_boxes(samples, height, width)
    output_golden = rocpycv.Tensor([samples, height, width, channels],
                                   rocpycv.eTensorLayout.NHWC, rocpycv.eDataType.U8, device)

    stream = rocpycv.Stream()
    output = rocpycv.bndbox(input, boxes, stream, device)
    rocpycv.bndbox_into(output_golden, input, boxes, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

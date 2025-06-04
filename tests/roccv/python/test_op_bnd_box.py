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

@pytest.mark.parametrize("input_path, device, expected_path, err", [
    ("test_input.bmp", rocpycv.GPU, "expected_bnd_box.bmp", 1.0),
    ("test_input.bmp", rocpycv.CPU, "expected_bnd_box.bmp", 1.0)
])

def test_op_remap(pytestconfig, input_path, device, expected_path, err):
    input_tensor = load_image(f"{pytestconfig.getoption('data_dir')}/{input_path}").copy_to(device)

    width = 720
    height = 480
    halfWidth = int(width/2)
    halfHeight = int(height/2)
    quarterWidth = int(width/4)
    quarterHeight = int(height/4)
    thirdWidth = int(width/3)
    thirdHeight = int(height/3)
    thirdDoubleHeight = int((height*2)/3)


    bbox_array = []
    numBoxes = []
    numBoxes.append(int(3))
    

    box1 = rocpycv.Box(quarterWidth, quarterHeight, halfWidth, halfHeight)
    box1_borderColor = rocpycv.Color4(0, 0, 255, 200)
    box1_fillColor = rocpycv.Color4(0, 255, 0, 100)
    bndbox1 = rocpycv.BndBox(box1, 5, box1_borderColor, box1_fillColor)

    box2 = rocpycv.Box(thirdWidth, thirdHeight, thirdWidth*2, quarterHeight)
    box2_borderColor = rocpycv.Color4(90, 16, 181, 50)
    box2_fillColor = rocpycv.Color4(0, 0, 0, 0)
    bndbox2 = rocpycv.BndBox(box2, -1, box2_borderColor, box2_fillColor)

    box3 = rocpycv.Box(-50, thirdDoubleHeight, width + 50, thirdHeight + 50)
    box3_borderColor = rocpycv.Color4(0, 0, 0, 50)
    box3_fillColor = rocpycv.Color4(111, 159, 232, 150)
    bndbox3 = rocpycv.BndBox(box3, 0, box3_borderColor, box3_fillColor)

    bbox_array.append(bndbox1)
    bbox_array.append(bndbox2)
    bbox_array.append(bndbox3)

    bnd_boxes = rocpycv.BndBoxes(1, numBoxes, bbox_array)


    stream = rocpycv.Stream()

    output_tensor = rocpycv.bndbox(input_tensor, bnd_boxes, stream, device)
    stream.synchronize()

    compare_image(output_tensor, f"{pytestconfig.getoption('data_dir')}/{expected_path}", err)
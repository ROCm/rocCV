# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

import numpy as np
import pytest
import rocpycv

# A 2x10x12x3 NHWC batch with distinct byte values, so every misplaced element is detected.
BASE = np.arange(2 * 10 * 12 * 3).astype(np.uint8).reshape(2, 10, 12, 3)

# Strided NumPy views that from_dlpack() wraps without copying (it keeps the producer's strides).
VIEWS = {
    "contiguous": lambda b: b,
    "w_crop": lambda b: b[:, :, 3:9],
    "h_crop": lambda b: b[:, 2:7],
    "roi": lambda b: b[:, 2:7, 3:9],
    "channel_slice": lambda b: b[..., :2],
    "w_step": lambda b: b[:, :, ::2],
    "n_step": lambda b: b[::2],
    "hw_transpose": lambda b: b.transpose(0, 2, 1, 3),
    "w_flip": lambda b: b[:, :, ::-1],
    "channel_flip": lambda b: b[..., ::-1],
    "single_channel": lambda b: b[..., 1:2],
}


@pytest.mark.parametrize("view", VIEWS.keys())
@pytest.mark.parametrize("device", [rocpycv.eDeviceType.CPU, rocpycv.eDeviceType.GPU])
def test_copy_to_strided_view(view, device):
    expected = VIEWS[view](BASE)
    tensor = rocpycv.from_dlpack(expected, rocpycv.NHWC)

    copied = tensor.copy_to(device)
    actual = np.from_dlpack(copied.copy_to(rocpycv.eDeviceType.CPU))

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("view", VIEWS.keys())
def test_copy_to_strided_view_float(view):
    base = np.arange(BASE.size, dtype=np.float32).reshape(BASE.shape)
    expected = VIEWS[view](base)
    tensor = rocpycv.from_dlpack(expected, rocpycv.NHWC)

    actual = np.from_dlpack(tensor.copy_to(rocpycv.eDeviceType.GPU).copy_to(rocpycv.eDeviceType.CPU))

    np.testing.assert_array_equal(actual, expected)

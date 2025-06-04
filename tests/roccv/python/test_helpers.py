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

import rocpycv
import numpy as np
import cv2


def load_image(image_path: str, grayscale: bool = False) -> rocpycv.Tensor:
    """Loads an image and returns a Tensor. This tensor will have U8 format and the NHWC layout.

    Args:
        image_path (str): Path to the image to load.
        grayscale (bool): Load image in grayscale mode. Defaults to False.

    Returns:
        rocpycv.Tensor: A tensor with the image contents loaded as data.
    """

    np_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED)
    assert np_array is not None
    if grayscale:
        # cv2 does not add a channel dimension if loaded as grayscale image.
        np_array = np.expand_dims(np_array, axis=-1)
    np_array = np.expand_dims(np_array, axis=0)
    return rocpycv.from_dlpack(np_array, rocpycv.NHWC)


def load_image_grayscale(image_path: str) -> rocpycv.Tensor:
    """Loads an image, converts it to grayscale and returns a Tensor. This tensor will have U8 format and the NHWC layout.

    Args:
        image_path (str): Path to the image to load.

    Returns:
        rocpycv.Tensor: A tensor with the image contents loaded as data.
    """

    img = cv2.imread(image_path)
    np_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert np_array is not None
    np_array = np.expand_dims(np_array, axis=0)
    np_array = np.expand_dims(np_array, axis=-1)
    return rocpycv.from_dlpack(np_array, rocpycv.NHWC)


def compare_image(tensor: rocpycv.Tensor, expected_path: str, error_threshold: float = 0.0) -> None:
    """Compares a rocpycv Tensor with an image.

    Args:
        tensor (rocpycv.Tensor): The tensor to compare against an image.
        expected_path (str): The path to the image to compare.
        error_threshold (float, optional): The maximum tolerated differences between pixels of the actual tensor and compared image. Defaults to 0.0.
    """
    expected = cv2.imread(expected_path, cv2.IMREAD_UNCHANGED)
    assert expected is not None
    expected = expected.flatten().tolist()

    compare_list(tensor, expected, error_threshold)


def compare_list(tensor: rocpycv.Tensor, expected: list, error_threshold: float = 0.0) -> None:
    """Compares tensor data with data given in a list.

    Args:
        tensor (rocpycv.Tensor): The tensor to compare against a list.
        expected (list): A list representing the expected data of the tensor.
        error_threshold (float, optional): The maximum tolerated differences between pixels of the actual tensor and compared image. Defaults to 0.0.
    """

    # Ensure the actual tensor is located on the CPU. We can do this using the copy_to method.
    actual_tensor = np.from_dlpack(tensor.copy_to(rocpycv.CPU))
    actual_tensor = actual_tensor.flatten().tolist()

    for i in range(len(expected)):
        err = abs(expected[i] - actual_tensor[i])
        if (err > error_threshold):
            raise Exception(f"Failed at index {i}, actual value {actual_tensor[i]} does not match {expected[i]}")

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


def rocpycv_type_to_np_type(type: rocpycv.eDataType) -> type:
    type_map = {
        rocpycv.eDataType.F32: np.float32,
        rocpycv.eDataType.F64: np.float64,
        rocpycv.eDataType.U8: np.uint8,
        rocpycv.eDataType.S8: np.int8,
        rocpycv.eDataType.U16: np.uint16,
        rocpycv.eDataType.S16: np.int16,
        rocpycv.eDataType.U32: np.uint32,
        rocpycv.eDataType.S32: np.int32,
    }

    if type not in type_map:
        raise RuntimeError("Cannot convert from specified rocpycv type to numpy type")

    return type_map[type]


def generate_tensor(samples: int, width: int, height: int, channels: int, dtype: rocpycv.eDataType, device: rocpycv.eDeviceType) -> rocpycv.Tensor:
    """Generate a rocpycv.Tensor with a NHWC layout containing random values on a specified device.

    Args:
        samples (int): Number of samples in the batch.
        width (int): Width of each image in the batch.
        height (int): Height of each image in the batch.
        channels (int): Number of channels for images in the batch.
        dtype (rocpycv.eDataType): Underlying datatype for the images.
        device (rocpycv.eDeviceType): Device this rocpycv.Tensor should be allocated on.

    Returns:
        rocpycv.Tensor: A rocpycv.Tensor containing randomly generated data.
    """
    return generate_tensor_generic([samples, height, width, channels], rocpycv.eTensorLayout.NHWC, dtype, device)


def generate_tensor_generic(shape: list[int], layout: rocpycv.eTensorLayout, dtype: rocpycv.eDataType, device: rocpycv.eDeviceType) -> rocpycv.Tensor:
    np_dtype = rocpycv_type_to_np_type(dtype)

    if np_dtype == np.float32 or np_dtype == np.float64:
        np_array = np.random.rand(*shape).astype(np_dtype)
    else:
        type_info = np.iinfo(np_dtype)
        np_array = np.random.randint(type_info.min, type_info.max, size=shape, dtype=np_dtype)

    tensor = rocpycv.from_dlpack(np_array, layout)
    return tensor.copy_to(device)


def compare_tensors(actual: rocpycv.Tensor, expected: rocpycv.Tensor) -> None:
    """Asserts that two tensors have the same metadata. The underlying data of the tensors are not compared.

    Args:
        actual (rocpycv.Tensor): The actual tensor resulting from the tests.
        expected (rocpycv.Tensor): The tensor which actual is expected to match with.
    """
    assert actual.shape() == expected.shape()
    assert actual.dtype() == expected.dtype()
    assert actual.layout() == expected.layout()
    assert actual.device() == expected.device()


def generate_random_array(shape: list[int], dtype: rocpycv.eDataType) -> np.ndarray:
    np_dtype = rocpycv_type_to_np_type(dtype)

    if np_dtype == np.float32 or np_dtype == np.float64:
        np_array = np.random.rand(*shape).astype(np_dtype)
    else:
        type_info = np.iinfo(np_dtype)
        np_array = np.random.randint(type_info.min, type_info.max, size=shape, dtype=np_dtype)

    return np_array


def compare_array(array: np.ndarray, expected_array: np.ndarray, diff_threshold: float = 0.0) -> None:
    # Ensure the actual tensor is located on the CPU. We can do this using the copy_to method.
    list = array.flatten().tolist()
    expected_list = expected_array.flatten().tolist()

    for i in range(len(expected_list)):
        diff = abs(expected_list[i] - list[i])
        if (diff > diff_threshold):
            raise Exception(f"Failed at index {i}, actual value {list[i]} does not match {expected_list[i]}")

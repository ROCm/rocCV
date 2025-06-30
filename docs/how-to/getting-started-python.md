# Getting Started (Python)
The following guide explains the basic use case of using the Python binding library for rocCV (called rocpycv).

## 1. Loading Data

`rocpycv` provides seamless support for DLPack, a common tensor data exchange format. This enables zero-copy conversions between `numpy` arrays and `rocpycv` tensors (as well as any other frameworks which support the DLPack protocol). Through DLPack, data can be shared efficiently without the overhead of memory duplication.

```python
import torch
import cv2
import rocpycv

# Load an image using OpenCV. Note, OpenCV loads colored images in the BGR order.
img = cv2.imread("image.png")

# Convert the image to RGB ordering
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the resulting numpy array into a rocpycv tensor
input_tensor = rocpycv.from_dlpack(img, rocpycv.eTensorLayout.HWC)
```

This will create a `rocpycv` tensor loaded on the host, as numpy only supports host tensors at the time of writing.

### 1.1) Moving tensor data from host to device
To move tensors from host to device, the `copy_to` method can be used.

```python
input_tensor_device = input_tensor.copy_to(rocpycv.eDeviceType.GPU)
```

## 2. Running operators

There are typically two ways to run operations on images in `rocpycv`:
1. `rocpycv.operator(...)` will return the resulting output tensor as a `rocpycv` tensor. In most cases, the shape of the output tensor will be inferred based on the type of the operation and the shape of the input parameters.
2. `rocpycv.operator_into(...)` expects the output tensor, manually created by the user, to be passed as a parameter into the operator. The device, shape, layout, and datatype must be manually created by the user and is subject to the limitations of that operator (described in detail in the rocCV C++ API reference).

```python
# Create a HIP stream to run device operators on
stream = rocpycv.Stream()

flip_code = 0 # a flip code of 0 will flip the image along the x-axis of the image
# Run the flip operator. If no device type is specified, the GPU version of the operator will be used.
output_tensor = rocpycv.flip(input_tensor_device, flip_code)

# Block until all work on the HIP stream has been completed
stream.synchronize()
```

> **Note:**
> When running operators on the GPU, they will be launched asynchronously and require explicit synchronization using a HIP stream. When running operators on the CPU, the calls are blocking and will only return once the operator has finished all work.

## 3. Exporting resulting image data
Since the DLPack protocol is supported with `rocpycv`, the resulting output tensor can be directly converted into a `torch` tensor to be used in machine learning workflows.

```python
# Creates a torch tensor from a rocpycv tensor
torch_tensor = torch.from_dlpack(output_tensor)

# Change the layout from HWC to CHW, as pytorch expects tensors to have this layout
torch_tensor = torch_tensor.permute(2, 0, 1)
```
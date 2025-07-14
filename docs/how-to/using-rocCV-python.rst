.. meta::
  :description: using the rocCV python library
  :keywords: rocCV, ROCm, library, python 

**************************************************
Using rocpycv, the rocCV Python binding library
**************************************************

rocCV is used for image pre- and post-processing. The rocCV library is a collection of computer vision operators. rocpycv are a collection of Python bindings for rocCV

This guide shows you how to set up a rocpycv project that runs the Flip operator.

This is the workflow that will be followed:

* Load image data into a tensor.
* Move the tensor data from the CPU to the GPU.
* Run the Flip operator.
* Export the output data.

Load image data into a tensor
===============================

rocpycv provides seamless support for the DLPack tensor data exchange format, allowing for zero-copy conversions between numpy arrays and rocpycv tensors. In zero-copy conversions, there isn't a need to save the data to CPU as an intermediate step.

The image is loaded into a numpy array using OpenCV, then converted into a rocpycv tensor using ``rocpycv.from_dlpack``.

.. code:: python

    import torch
    import cv2
    import rocpycv

    # Load an image using OpenCV. Note, OpenCV loads colored images in the BGR order.
    img = cv2.imread("image.png")

    # Convert the image to RGB ordering
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the resulting numpy array into a rocpycv tensor
    input_tensor = rocpycv.from_dlpack(img, rocpycv.eTensorLayout.HWC)


Because numpy only supports host tensors, the rocpycv tensor will always be created on the CPU. Use ``copy_to`` to move the tensor from the CPU to the GPU:

.. code:: python

    input_tensor_device = input_tensor.copy_to(rocpycv.eDeviceType.GPU)


Run the Flip operator 
=======================

There are two ways to run operators:

1. Using the rocpycv operator method. This method returns the output tensor as a rocpycv tensor. The shape of the output tensor is inferred from the operator and its input parameters. 
2. Using the rocpycv operator_into method. The output tensor is passed as a parameter, and the device, shape, layout, and datatype must be set when the tensor is constructed.

You can specify whether to run the operator on the GPU or the CPU. If a device type isn't specified, the operator will run on the GPU by default.

When the operator runs on CPU, the operator will block until the operation has completed. However, when the operator runs on GPU, the call to the operator is non-blocking and external synchronization is needed.

In this example, the Flip operator is run on GPU with external synchronization using ``rocpycv.flip()``:


.. code:: python

    # Create a HIP stream to run device operators on
    stream = rocpycv.Stream()

    flip_code = 0 # a flip code of 0 flips the image along its x-axis
    # Run the flip operator. If no device type is specified, the GPU version of the operator will be used.
    output_tensor = rocpycv.flip(input_tensor_device, flip_code)

    # Block until all work on the HIP stream has been completed
    stream.synchronize()


Exporting the output data to memory
====================================

The output tensor can either be passed to another operator for further processing or it can be moved back into host (CPU) memory.

rocpycv supports the DLPack protocol and the resulting output tensor can be directly converted into a torch tensor to be used in machine learning workflows.

.. code:: python

    # Create a torch tensor from a rocpycv tensor
    torch_tensor = torch.from_dlpack(output_tensor)

    # Change the layout from HWC to CHW, as pytorch expects tensors to have this layout
    torch_tensor = torch_tensor.permute(2, 0, 1)
    
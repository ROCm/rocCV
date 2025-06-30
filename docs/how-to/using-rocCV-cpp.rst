.. meta::
  :description: using the rocCV C++ library
  :keywords: rocCV, ROCm, API,  C++, cpp, api 

******************************************
Using rocCV with C++ 
******************************************

rocCV is used for image pre- and post-processing. The rocCV library is a collection of computer vision operators. 

This guide shows you how to set up a rocCV project that runs the Flip operator.

This is the workflow that will be followed:

* Create both the input and output tensors using the tensor requirements.
* Decode the image and load its data into the input tensor.
* Pass the input and output tensors to the Flip operator.
* Move the data from the output tensor to memory.

Construct input and output tensors
===================================

Both the input and output tensors are constructed using the tensor requirements. 

In this example, the tensor requirements are calculated using ``CalcRequirements`` and ``roccv::ImageFormat``. 

``roccv::ImageFormat`` specifies an NHWC format for the tensors. The NHWC format uses the number of images in the batch, the dimensions of each image in the batch, and the number of channels to represent a tensor. The format is then passed to ``CalRequirements``

``roccv::ImageFormat`` uses pre-defined formats. Overloads of ``roccv::Tensor::CalcRequirements()`` can be used to customize the shape, layout, and datatype of a tensor. 

.. code:: cpp

  int numImages = 1;      // Number of images in the batch
  int imageWidth = 720;   // Width for each image in the batch
  int imageHeight = 480;  // Height for each image in the batch

  roccv::ImageFormat inputFmt = roccv::FMT_RGB8;  // For 3-channel, 8-bit interleaved RGB images

  // Calculate tensor requirements
  roccv::Tensor::Requirements reqs = roccv::Tensor::CalcRequirements(numImages, {imageWidth, imageHeight}, inputFmt, eDeviceType::GPU);

  roccv::Tensor input(reqs);
  roccv::Tensor output(reqs);

.. note::
  
  In this example the input and the output tensors are assumed to have the same requirements. However, this isn't the case generally, and tensor requirements of the input and output tensors will have to be calculated separately.

Move image data into the input tensor
====================================

Once the input tensor is created, the decoded image data can be moved into it. 

``hipMemcpy`` and ``hipMemcpyAsync`` are used if GPU memory is being used, and ``memcpy`` is used if CPU memory is being used. 

``exportData`` provides a way to access the raw data in the tensors through ``basePtr()```.

.. code:: cpp

  unsigned char *imageData; // Decoded image data 
  
  // Export tensor data from the input tensor to get access to the raw data.
  auto tensorData = input.exportData<roccv::TensorDataStrided>();

  // Move the decoded image data into the input tensor. In this case, the input tensor memory is on the GPU.
  
  size_t imageSizeBytes = input.shape().size() * input.dtype().size();
  hipMemcpy(tensorData.basePtr(), imageData, imageSizeBytes);

Call the Flip operator 
=======================

Both the input and output tensors are passed to the operator. 

You can specify whether to run the operator on the GPU (``eDeviceType::GPU``) or the CPU (``eDeviceType::CPU``). If a device type isn't specified, the operator will run on the GPU by default.

When the operator runs on CPU, the operator will block until the operation has completed. However, when the operator runs on GPU, the call to the operator is non-blocking and external synchronization is needed.

In this example, the operator is run on GPU with external synchronization.

.. code:: cpp 

  // Create a HIP stream for the operator.
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Specify operator parameters
  int32_t flipCode = 0;   // Flip code 0 will flip the image along the X-axis.

  // Create and call the flip operator on the newly created HIP stream with the GPU.
  roccv::Flip flip;
  flip(stream, input, output, flipCode, eDeviceType::GPU);
  // Can also call the GPU version of the operator by default using: flip(stream, input, output, flipCode)

  // Optionally, additional operators can be queued up on the same stream to create an image processing pipeline.

  // Block until all work on the provided stream has been completed, and destroy the stream once finished.
  hipStreamSynchronize(stream);
  hipStreamDestroy(stream);

Exporting the output data to memory
====================================

The results of the Flip operation are written to the output tensor. The output tensor can either be passed into another operator to further process the image data or it can be moved back into host (CPU) memory.

In this example, the data will be exported to memory using ``exportData``.

.. code:: cpp

  // Allocate memory on the host to move output data into
  std::vector<unsigned char> outputHost(output.shape().size());

  // Export output tensor data and move to host allocated memory
  auto outputTensorData = output.exportData<roccv::TensorDataStrided>();
  hipMemcpy(outputHost.data(), outputTensorData.basePtr(), output.shape().size() * output.dtype().size(), hipMemcpyDeviceToHost);
   
  
# rocCV Samples

## Description
The rocCV samples demonstrate the use of the operator interfaces in C++ and Python to process images and construct image processing pipelines. The code shows how to set up tensors, load images into the tensors from OpenCV mat objects and then set up and run the operators.

## Operator Samples
1. Individual process in C++: bnd_box.cpp, center_crop.cpp, composite.cpp, copy_make_border.cpp, custom_crop.cpp, gamma_contrast.cpp, normalize.cpp, warp_perspective.cpp.
2. cropandresize - Crops and resizes the input image. This sample is designed to demonstrate a simple pipeline for multiple operators.
3. pipeline/multi_op_1.py: A pipeline of color conversion, cropping, bilateral filtering, bounding box drawing, rotation and resizing.

## Building and running the samples
Build rocCV as described in the main README with the `-D SAMPLES=ON` flag set.
Samples will be available in the build/bin directory.
Run command option "-h" for usage.
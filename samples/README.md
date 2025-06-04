# rocCV Samples

## Description
The rocCV samples available here show how to set up and run several of the operators available in rocCV.
The code shows how to set up tensors, load images into the tensors from OpenCV mat objects and then set up and run the operators.

## Operator Samples
1. BndBox - Draws bounding boxes over the input image
2. Composite - Composites two images together.
3. Copy Make Border - Creates a border around the input image.
4. Gamma Contrast - Adjusts the gamma contrast of the image. This example gives two outputs with different gamma correction values applied.
5. Normalize - Normalizes the pixel range of the input image.
6. Warp Perspective - Applies a perspective transformation on the input image.
7. Crop and Resize - Crops and resizes the input image. This sample is designed to demonstrate a simple pipeline for multiple operators.

## Building and running the samples
Build rocCV as described in the main README with the `-D SAMPLES=ON` flag set.
Samples will be available in the build/bin directory.

An image for the samples can be found at data/samples/python/input/test_input_0.bmp

#### Example for running BndBox
./bnd_box ../../data/samples/python/input/test_input_0.bmp bndBox_out.bmp 0

This will create an output image called bndBox_out.bmp with bounding boxes drawn over the input image.

#### Example for running Gamma Contrast
./gamma_contrast ../../data/samples/python/input/test_input_0.bmp gamma_contrast_1.bmp gamma_contrast_2.bmp

This will create gamma_contrast_1.bmp and gamma_contrast_2.bmp which will both have different gamma correction applied to them.

#### Example for running Crop and Resize
./roccv_cropandresize_app -i ../../data/samples/python/input/test_input_0.bmp -b 1

This will output a cropped and resized input image.
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

import os
import math
import numpy as np
import rocpycv
import cv2
import argparse

"""This simple demonstrates the useage of multiple operators in the image processing pipeline.
"""


def load_image_batch(directory: str) -> np.ndarray:
    """Loads images within a directory to a numpy array in the NHWC layout. Ensure all images within this directory are of the same size.

    Args:
        directory (str): The directory containing images to load.

    Raises:
        FileNotFoundError: Raised when an invalid directory is passed or an image does not exist.

    Returns:
        np.ndarray: A numpy array containing the loaded batch of images.
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        image_path = os.path.join(directory, filename)

        # Ensure that we only read files which are supported
        _, extension = os.path.splitext(image_path)
        if (extension not in [".png", ".bmp", ".jpg", ".jpeg"]):
            continue

        image = cv2.imread(image_path)
        # Check that we were able to read the image properly.
        if (image is None):
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        # Convert the default BGR to YUV
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        #images.append(image)
        images.append(yuv_image)
    return np.stack(images)

def write_image_batch(image_batch: np.ndarray, output_directory: str) -> None:
    """Writes a batch of images from a numpy array to disk.

    Args:
        image_batch (np.ndarray): A numpy array with a NHWC layout. 
        output_directory (str): The directory to output images to.
    """
    for i, image in enumerate(image_batch):
        cv2.imwrite(os.path.join(output_directory, f"output_{i}.png"), image)

def calc_center_shift(center_x, center_y, angle) -> tuple[float, float]:
    x_shift = (1 - math.cos(angle * math.pi / 180)) * center_x - math.sin(angle * math.pi / 180) * center_y
    y_shift = math.sin(angle * math.pi / 180) * center_x + (1 - math.cos(angle * math.pi / 180)) * center_y
    return (x_shift, y_shift)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Path to a directory containing images", required=True)
    parser.add_argument("--output_dir", help="Path to the image output directory", default=".")
    args = parser.parse_args()

    # To convert from a numpy array to a rocpycv tensor, we use rocpycv.to_dlpack and provide a layout.
    input_tensor_yuv = rocpycv.from_dlpack(load_image_batch(args.input_dir), rocpycv.NHWC)
    # Ensure this tensor is on the GPU, since we want to perform our operations on the GPU
    input_tensor_yuv = input_tensor_yuv.copy_to(rocpycv.GPU)

    # Create a stream to run the operations on, and then queue up operations on this stream.
    stream = rocpycv.Stream()

    # Color convert from YUV to BGR
    input_tensor = rocpycv.cvtcolor(input_tensor_yuv, rocpycv.COLOR_YUV2BGR, stream, rocpycv.GPU)

    # Crop
    input_shape = input_tensor.shape()
    crop_x = input_shape[2] >> 2
    crop_y = input_shape[1] >> 2
    crop_w = input_shape[2] >> 1
    crop_h = input_shape[1] >> 1
    cropped_tensor = rocpycv.custom_crop(input_tensor, rocpycv.Box(crop_x, crop_y, crop_w, crop_h), stream, rocpycv.GPU)

    # Bilateral filter
    filtered_tensor = rocpycv.bilateral_filter(cropped_tensor, 30, 100, 100, rocpycv.REPLICATE, [0, 0, 0, 0], stream, rocpycv.GPU)

    # Bounding box
    shape = filtered_tensor.shape()
    bbox_array = []
    numBoxes = []
    numBoxes.append(int(1))
    
    box1 = rocpycv.Box((shape[2] >> 1), (shape[1] >> 1), (shape[2] >> 2), (shape[1] >> 3))
    box1_borderColor = rocpycv.Color4(0, 0, 255, 200)
    thickness = 8
    box1_fillColor = rocpycv.Color4(0, 0, 0, 0)
    bndbox1 = rocpycv.BndBox(box1, thickness, box1_borderColor, box1_fillColor)
    bbox_array.append(bndbox1)

    bnd_boxes = rocpycv.BndBoxes(1, numBoxes, bbox_array)

    box_tensor = rocpycv.bndbox(filtered_tensor, bnd_boxes, stream, rocpycv.GPU)

    # Rotate
    angle = 180
    center_x = math.floor((shape[2] + 1) / 2)
    center_y = math.floor((shape[1] + 1) / 2)
    shift = calc_center_shift(center_x, center_y, angle)
    rotated_tensor = rocpycv.rotate(box_tensor, angle, shift, rocpycv.eInterpolationType.LINEAR, stream, rocpycv.GPU)

    # Resize
    # For the resized image batch shape, batch size and channels remain the same, but we adjust the width and height of the final image.
    output_shape = input_tensor.shape()
    output_shape = tuple(output_shape)
    resized_tensor = rocpycv.resize(rotated_tensor, output_shape, rocpycv.LINEAR, stream, rocpycv.GPU)

    # Ensure all work on the stream has been finished before writing the image batch back to disk.
    stream.synchronize()

    if (not os.path.exists(args.output_dir)):
        raise FileNotFoundError(f"Directory does not exist: {args.output_dir}")

    # Copy the final tensor back to the host (CPU) so that it can be converted back into a numpy array and write images
    # back to disk.
    resized_array = np.from_dlpack(resized_tensor.copy_to(rocpycv.CPU))
    write_image_batch(resized_array, args.output_dir)

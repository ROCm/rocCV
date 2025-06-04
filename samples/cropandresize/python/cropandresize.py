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
import numpy as np
import rocpycv
import cv2
import argparse

"""This simple crop and resize sample loads in a batch of images from a directory and demonstrates how
operators can be used with the rocpycv library.
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
        images.append(image)
    return np.stack(images)


def write_image_batch(image_batch: np.ndarray, output_directory: str) -> None:
    """Writes a batch of images from a numpy array to disk.

    Args:
        image_batch (np.ndarray): A numpy array with a NHWC layout. 
        output_directory (str): The directory to output images to.
    """
    for i, image in enumerate(image_batch):
        cv2.imwrite(os.path.join(output_directory, f"output_{i}.png"), image)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Path to a directory containing images", required=True)
    parser.add_argument("--output-dir", help="Path to the image output directory", default=".")
    args = parser.parse_args()

    # To convert from a numpy array to a rocpycv tensor, we use rocpycv.to_dlpack and provide a layout.
    input_tensor = rocpycv.from_dlpack(load_image_batch(args.input_dir), rocpycv.NHWC)
    # Ensure this tensor is on the GPU, since we want to perform our operations on the GPU
    input_tensor = input_tensor.copy_to(rocpycv.GPU)

    # For the resized image batch shape, batch size and channels remain the same, but we adjust the width and height of the final image.
    output_shape = input_tensor.shape()
    output_shape[1] = 100
    output_shape[2] = 100
    output_shape = tuple(output_shape)

    # Create a stream to run the operations on, and then queue up operations on this stream.
    stream = rocpycv.Stream()
    cropped_tensor = rocpycv.custom_crop(input_tensor, rocpycv.Box(50, 50, 400, 400), stream, rocpycv.GPU)
    resized_tensor = rocpycv.resize(cropped_tensor, output_shape, rocpycv.CUBIC, stream, rocpycv.GPU)

    # Ensure all work on the stream has been finished before writing the image batch back to disk.
    stream.synchronize()

    if (not os.path.exists(args.output_dir)):
        raise FileNotFoundError(f"Directory does not exist: {args.output_dir}")

    # Copy the final tensor back to the host (CPU) so that it can be converted back into a numpy array and write images
    # back to disk.
    resized_tensor = np.from_dlpack(resized_tensor.copy_to(rocpycv.CPU))
    write_image_batch(resized_tensor, args.output_dir)

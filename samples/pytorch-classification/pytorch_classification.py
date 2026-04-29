# ##############################################################################
# Copyright (c)  - 2026 Advanced Micro Devices, Inc.
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

"""Classification with rocCV preprocessing and PyTorch inference"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from torchvision import models as torchvision_models
import cv2
import rocpycv


def read_image(image_path: str) -> np.ndarray:
    """Read an image from disk and convert to numpy array.

    Args:
        image_path: Path to the image file

    Returns:
        NumPy array in HWC format with uint8 data type
    """
    # Read image using OpenCV (BGR format)
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return rgb_image


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classification with rocCV preprocessing and PyTorch inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image",
    )
    return parser.parse_args()


def main() -> None:
    """Classification with rocCV preprocessing and PyTorch inference."""
    args: argparse.Namespace = parse_args()

    # 1. Load PyTorch ResNet50 model
    print("Loading PyTorch ResNet50 model...")
    weights = torchvision_models.ResNet50_Weights.IMAGENET1K_V1
    resnet50_base = torchvision_models.resnet50(weights=weights)
    resnet50_base.eval()

    # Wrap with softmax layer for probability output
    class Resnet50_Softmax(torch.nn.Module):
        def __init__(self, resnet50):
            super(Resnet50_Softmax, self).__init__()
            self.resnet50 = resnet50

        def forward(self, x):
            infer_output = self.resnet50(x)
            return torch.nn.functional.softmax(infer_output, dim=1)

    model = Resnet50_Softmax(resnet50_base)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. Read the image
    print(f"Reading image: {args.input}")
    np_image = read_image(args.input)
    print(f"Input image shape: {np_image.shape}")

    # 3. Preprocess using rocCV operations
    print(f"Preprocessing with rocCV...")

    # 3.1 Convert numpy array to rocCV tensor
    # Add batch dimension using stack
    np_image_batch = np.stack([np_image])
    input_tensor = rocpycv.from_dlpack(np_image_batch, rocpycv.NHWC)
    input_tensor = input_tensor.copy_to(rocpycv.GPU)

    stream = rocpycv.Stream()

    # 3.2 Resize to target dimensions
    output_shape = (1, 224, 224, 3)  # NHWC format
    resized_tensor = rocpycv.resize(input_tensor, output_shape, rocpycv.LINEAR, stream, rocpycv.GPU)

    # 3.3 Convert to float32 WITHOUT scaling
    float_tensor = rocpycv.convert_to(resized_tensor, rocpycv.eDataType.F32, 1.0, 0.0, stream, rocpycv.GPU)

    # 3.4 Normalize with ImageNet mean and std
    # Scale mean and std to [0, 255] range to match float_tensor
    # Mathematically: (pixel - mean*255) / (std*255) == (pixel/255 - mean) / std
    mean_scaled = torch.Tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])  # [123.675, 116.28, 103.53]
    mean_torch = mean_scaled.reshape(1, 1, 1, 3).cuda()
    mean_tensor = rocpycv.from_dlpack(mean_torch, rocpycv.eTensorLayout.NHWC)

    std_scaled = torch.Tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])  # [58.395, 57.12, 57.375]
    std_torch = std_scaled.reshape(1, 1, 1, 3).cuda()
    std_tensor = rocpycv.from_dlpack(std_torch, rocpycv.eTensorLayout.NHWC)

    # Normalize: output = (input - mean) / std
    normalized_tensor = rocpycv.normalize(
        float_tensor,
        mean_tensor,
        std_tensor,
        rocpycv.NormalizeFlags.SCALE_IS_STDDEV,
        1.0,  # global_scale
        0.0,  # global_shift
        0.0,  # epsilon (to avoid division by zero)
        stream,
        rocpycv.GPU
    )

    reformatted_tensor = rocpycv.reformat(normalized_tensor, rocpycv.eTensorLayout.NCHW, stream, rocpycv.GPU)
    stream.synchronize()

    # 4. Convert rocCV tensor to PyTorch tensor
    print("Converting to PyTorch tensor...") 
    torch_tensor = torch.from_dlpack(reformatted_tensor)

    print(f"PyTorch tensor shape (NCHW): {torch_tensor.shape}")

    # 5. Run PyTorch inference
    print("Running inference...")
    with torch.no_grad():
        probabilities = model(torch_tensor)

    # 6. Postprocess the inference results
    probabilities_cpu = probabilities.cpu().numpy()

    # 7. Print the top 5 predictions
    print("\nTop 5 predictions:")
    indices = np.argsort(probabilities_cpu)[0][::-1]
    for i, index in enumerate(indices[:5]):
        print(f"  {i+1}. Class {index}: {probabilities_cpu[0][index]:.6f}")


if __name__ == "__main__":
    main()

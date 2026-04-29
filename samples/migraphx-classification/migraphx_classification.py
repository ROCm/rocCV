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

"""Classification with rocCV preprocessing and MIGraphX inference."""

from __future__ import annotations

import argparse
import os

import cv2
import migraphx
import numpy as np
import rocpycv


# ImageNet normalization, scaled to the [0, 255] pixel range so we can apply
# them directly to U8-derived float pixels without a separate /255 step:
#   (pixel/255 - mean) / std  ==  (pixel - mean*255) / (std*255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

INPUT_H, INPUT_W = 224, 224


def read_image(image_path: str) -> np.ndarray:
    """Read an image from disk as an NHWC uint8 BGR numpy array."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return np.stack([bgr])


def load_or_compile_model(onnx_path: str) -> migraphx.program:
    """Load a cached compiled model, or parse + compile + cache the ONNX file."""
    # TODO: Support other batch sizes later
    batch_size = 1
    cache_path = f"{os.path.splitext(onnx_path)[0]}_b{batch_size}.mxr"

    if os.path.exists(cache_path):
        print(f"Loading cached compiled model: {cache_path}")
        return migraphx.load(cache_path, format="msgpack")

    print(f"Parsing ONNX: {onnx_path}")
    model = migraphx.parse_onnx(
        onnx_path,
        map_input_dims={"data": [batch_size, 3, INPUT_H, INPUT_W]},
    )

    print("Compiling for GPU...")
    # offload_copy=False allows us to bind GPU buffers directly to allow for
    # zero-copy interop.
    model.compile(migraphx.get_target("gpu"), offload_copy=False)

    print(f"Caching compiled model to: {cache_path}")
    migraphx.save(model, cache_path, format="msgpack")
    return model


def load_labels(labels_path: str | None) -> list[str] | None:
    if labels_path is None:
        return None
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classification with rocCV preprocessing and MIGraphX inference"
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--model", required=True, help="Path to a ResNet50 ONNX model"
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional path to a newline-separated ImageNet class labels file",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    args = parse_args()

    # 1. Load the model
    model = load_or_compile_model(args.model)

    print(f"Reading image: {args.input}")
    np_image = read_image(args.input)
    print(f"Input image shape: {np_image.shape}")

    print("Preprocessing with rocCV...")
    stream = rocpycv.Stream()

    # 2. Convert the image to a rocCV tensor in NHWC layout.
    tensor = rocpycv.from_dlpack(np_image, rocpycv.NHWC).copy_to(rocpycv.GPU)

    # 3. Convert from BGR to RGB for MIGraphX.
    tensor = rocpycv.cvtcolor(tensor, rocpycv.COLOR_BGR2RGB, stream, rocpycv.GPU)

    # 4. Resize to 224x224.
    tensor = rocpycv.resize(tensor, (1, INPUT_H, INPUT_W, 3), rocpycv.CUBIC, stream, rocpycv.GPU)

    # 5. Cast U8 -> F32 (no scaling; normalize step folds in /255).
    tensor = rocpycv.convert_to(tensor, rocpycv.eDataType.F32, 1.0, 0.0, stream, rocpycv.GPU)

    # 6. ImageNet normalize: (pixel - mean) / std.
    mean_t = rocpycv.from_dlpack(IMAGENET_MEAN.reshape(1, 1, 1, 3), rocpycv.NHWC).copy_to(rocpycv.GPU)
    std_t = rocpycv.from_dlpack(IMAGENET_STD.reshape(1, 1, 1, 3), rocpycv.NHWC).copy_to(rocpycv.GPU)
    tensor = rocpycv.normalize(tensor, mean_t, std_t, rocpycv.NormalizeFlags.SCALE_IS_STDDEV, 1.0, 0.0, 0.0, stream, rocpycv.GPU)

    # 7. NHWC -> NCHW (MIGraphX / ONNX expects NCHW).
    tensor = rocpycv.reformat(tensor, rocpycv.eTensorLayout.NCHW, stream, rocpycv.GPU)
    print(f"Preprocessed tensor shape (NCHW): {tensor.shape()}")

    print("Running MIGraphX inference...")

    # Setup MIGraphX arguments/shapes
    in_shape = migraphx.shape(type="float_type", lens=tensor.shape())
    out_shape = migraphx.shape(type="float_type", lens=[1, 1000])
    in_arg = migraphx.argument_from_pointer(in_shape, tensor.data_ptr())
    out_buf = migraphx.allocate_gpu(out_shape)

    outputs = model.run_async(
        {"data": in_arg, "main:#output_0": out_buf},
        stream.handle(),
        "ihipStream_t",
    )
    stream.synchronize()

    logits = np.array(migraphx.from_gpu(outputs[0]))
    probs = softmax(logits, axis=1)

    labels = load_labels(args.labels)

    # Report top-K for the first image in the batch.
    print(f"\nTop {args.top_k} predictions:")
    top = np.argsort(probs[0])[::-1][: args.top_k]
    for rank, idx in enumerate(top, start=1):
        name = labels[idx] if labels is not None and idx < len(labels) else f"class {idx}"
        print(f"  {rank}. {name}: {probs[0][idx]:.6f}")


if __name__ == "__main__":
    main()

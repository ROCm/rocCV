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
BATCH_SIZE = 1


def read_image(image_path: str) -> np.ndarray:
    """Read an image from disk as an NHWC uint8 BGR numpy array."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return np.stack([bgr])


def load_or_compile_model(onnx_path: str, use_fp16: bool = True) -> migraphx.program:
    """Load a cached compiled model, or parse + compile + cache the ONNX file."""
    precision_tag = "fp16" if use_fp16 else "fp32"
    cache_path = f"{os.path.splitext(onnx_path)[0]}_b{BATCH_SIZE}_{precision_tag}.mxr"

    if os.path.exists(cache_path):
        print(f"Loading cached compiled model: {cache_path}")
        return migraphx.load(cache_path, format="msgpack")

    print(f"Parsing ONNX: {onnx_path}")
    model = migraphx.parse_onnx(
        onnx_path,
        map_input_dims={"data": [BATCH_SIZE, 3, INPUT_H, INPUT_W]},
    )

    if use_fp16:
        print("Quantizing to FP16...")
        # Inserts internal float -> half conversions; model inputs/outputs stay
        # float32, so the existing F32 buffer setup remains unchanged.
        migraphx.quantize_fp16(model)

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

    model = load_or_compile_model(args.model)

    print(f"Reading image: {args.input}")
    np_image = read_image(args.input)
    print(f"Input image shape: {np_image.shape}")

    # Load/allocate tensors on the GPU
    input_tensor  : rocpycv.Tensor = rocpycv.from_dlpack(np_image, "NHWC").copy_to(rocpycv.GPU)
    resized       : rocpycv.Tensor = rocpycv.Tensor((BATCH_SIZE, INPUT_H, INPUT_W, 3), "NHWC", np.uint8)
    rgb           : rocpycv.Tensor = rocpycv.Tensor((BATCH_SIZE, INPUT_H, INPUT_W, 3), "NHWC", np.uint8)
    f32           : rocpycv.Tensor = rocpycv.Tensor((BATCH_SIZE, INPUT_H, INPUT_W, 3), "NHWC", np.float32)
    normalized    : rocpycv.Tensor = rocpycv.Tensor((BATCH_SIZE, INPUT_H, INPUT_W, 3), "NHWC", np.float32)
    preprocessed  : rocpycv.Tensor = rocpycv.Tensor((BATCH_SIZE, 3, INPUT_H, INPUT_W), "NCHW", np.float32)

    mean_t        : rocpycv.Tensor = rocpycv.from_dlpack(IMAGENET_MEAN.reshape(1, 1, 1, 3), "NHWC").copy_to(rocpycv.GPU)
    std_t         : rocpycv.Tensor = rocpycv.from_dlpack(IMAGENET_STD.reshape(1, 1, 1, 3), "NHWC").copy_to(rocpycv.GPU)

    # Setup MIGraphX arguments/shapes
    in_shape   : migraphx.shape    = migraphx.shape(type="float_type", lens=preprocessed.shape())
    out_shape  : migraphx.shape    = migraphx.shape(type="float_type", lens=[BATCH_SIZE, 1000])

    in_arg     : migraphx.argument = migraphx.argument_from_pointer(in_shape, preprocessed.data_ptr())
    out_buf    : migraphx.buffer   = migraphx.allocate_gpu(out_shape)

    # Begin preprocessing
    print("Preprocessing with rocCV...")
    stream : rocpycv.Stream = rocpycv.Stream()

    rocpycv.resize_into(resized, input_tensor, rocpycv.CUBIC, stream)
    rocpycv.cvtcolor_into(rgb, resized, rocpycv.COLOR_BGR2RGB, stream)
    rocpycv.convert_to_into(f32, rgb, 1.0, 0.0, stream)
    rocpycv.normalize_into(normalized, f32, mean_t, std_t, rocpycv.NormalizeFlags.SCALE_IS_STDDEV, 1.0, 0.0, 0.0, stream)
    rocpycv.reformat_into(preprocessed, normalized, stream)
    
    print(f"Preprocessed tensor shape (NCHW): {preprocessed.shape()}")

    print("Running MIGraphX inference...")

    outputs = model.run_async(
        {"data": in_arg, "main:#output_0": out_buf},
        stream.handle(),
        "ihipStream_t",
    )
    stream.synchronize()

    # Postprocess the inference results
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

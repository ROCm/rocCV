# rocCV MIGraphX Classification Sample

This sample demonstrates how to use rocCV to preprocess an image on the GPU and run inference with a ResNet50 ONNX model through MIGraphX. The preprocessed tensor is handed off to MIGraphX via a raw GPU pointer for zero-copy interop, so no host round-trip is needed between preprocessing and inference.

## Dependencies

- A rocCV build with the Python bindings (`rocpycv`) on `PYTHONPATH`. Build rocCV with Python 3.11 by passing the following to cmake:
  ```shell
  -DPYTHON_VERSION_SUGGESTED=3.11
  ```
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX) with its Python bindings.
- `opencv-python` and `numpy`.
- A ResNet50 ONNX model with input name `data` and shape `[N, 3, 224, 224]` (e.g. the ONNX Model Zoo `resnet50-v1-7.onnx`).
- Optional: a newline-separated ImageNet class labels file for human-readable output.

## Command line

```shell
python3.11 migraphx_classification.py \
    --input  path/to/image.jpg \
    --model  path/to/resnet50.onnx \
    --labels path/to/imagenet_classes.txt \
    --top-k  5
```

Arguments:
- `--input` (required): path to the input image.
- `--model` (required): path to the ResNet50 ONNX file.
- `--labels` (optional): path to an ImageNet class label file. If omitted, classes are reported by index.
- `--top-k` (optional, default 5): number of top predictions to print.

On the first run, the script compiles the ONNX model for the GPU and caches the result alongside the ONNX file as `<model>_b1.mxr`. Subsequent runs load the cached `.mxr` directly and skip compilation.

## Preprocessing Operators

The preprocessing pipeline runs entirely on the GPU through `rocpycv`:

1. **CvtColor**: Converts the OpenCV BGR image to RGB.
2. **Resize**: Resizes to 224x224 using cubic interpolation.
3. **Convert To**: Casts U8 pixels to float32 (no scaling — the `/255` step is folded into the normalize parameters).
4. **Normalize**: Applies ImageNet mean/std normalization. The mean and std constants are pre-multiplied by 255 so the operator can normalize directly from the [0, 255] float pixel range in a single pass.
5. **Reformat**: Converts the tensor from NHWC to NCHW, the layout MIGraphX/ONNX expects.

## MIGraphX Interop

The compiled MIGraphX program is built with `offload_copy=False`, so input and output buffers must already live on the GPU. The sample binds:
- The rocCV preprocessed tensor's GPU pointer (`tensor.data_ptr()`) as the `data` input via `migraphx.argument_from_pointer`.
- A `migraphx.allocate_gpu` buffer as the output.

Inference is launched with `model.run_async` using the same HIP stream as the preprocessing pipeline (`stream.handle()`), so preprocessing and inference are serialized on a single stream with no extra synchronization until the final `stream.synchronize()`.

The output logits are copied back to the host with `migraphx.from_gpu`, passed through softmax, and the top-K classes are printed.

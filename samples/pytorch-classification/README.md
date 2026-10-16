# rocCV PyTorch Classification Sample
This sample demonstrates how to use rocCV to preprocess an image for classification and run it through the ResNet50 model with PyTorch.

## Dependencies
Build rocCV with Python version 3.11 by giving the following in the cmake command:
```shell
-DPYTHON_VERSION_SUGGESTED=3.11
```

## Command line
```shell
python3.11 pytorch_classification.py --input path/to/image
```

## Preprocessing Operators
1. Resize: Resizes the input image to 224x224.
2. Convert To: Converts the input pixels to float32.
3. Normalize: Normalizes using ImageNet Statistics.
4. Reformat: Converts the tensor from NHWC to NCHW which is the format expected by PyTorch.
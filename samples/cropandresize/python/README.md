# rocpycv Crop and Resize Sample
## Overview
This sample demonstrates a simple pipeline which demonstrates the following functionality:
* Loading images into a `rocpycv` Tensor using `cv2` and `numpy`.
* Moving a Tensor from the host/device.
* Creating a stream.
* Running the `custom_crop` and `resize` operations on a created stream.
* Writing images back to disk by converting a `rocpycv` Tensor back into a `numpy` array.
## Usage
```bash
python3 cropandresize.py --input-dir INPUT_DIR [--output-dir OUTPUT_DIR]
```
* `--input-dir`: A directory containing images to be added to the batch. Note that all images within this batch must be of the same size, as only constant size Tensors are supported at this point in time.
* `--output-dir`: Represents the output directory images will be written to. All images will be written as `output_{index}.png`.
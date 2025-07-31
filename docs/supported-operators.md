# Supported Operators

See below for a list of Computer Vision operators rocCV supports.

|Name|Description|Datatypes|Layouts|CPU/GPU Support|
|-|-|-|-|-|
|BilateralFilter|Applies a bilateral filter.|U8|NHWC, HWC|Both|
|BndBox|Draws bounding boxes on the images in a tensor.|U8|NHWC, HWC|Both|
|CenterCrop|Crops an image at its center with a given rectangular region.|U8, S8, U16, S16, U32, S32, F32, F64|NHWC, HWC|Both|
|Composite|Composites two input tensors using a provided alpha mask.|U8, S8, U32, S32, F32|NHWC, HWC|Both|
|CopyMakeBorder|Generates a border using a specified border mode around the input tensor.|U8, S8, U32, S32, F32|NHWC|Both|
|CustomCrop|Crops an image with a given rectangular region.|U8, S8, U16, S16, U32, S32, F32, F64|NHWC, HWC|Both|
|CvtColor|Converts the color space of the images in a tensor.|U8|NHWC, HWC|Both|
|Flip|Flips the images in a tensor about the horizontal, vertical or both axes.|U8, S32, F32|NHWC, HWC|Both|
|GammaContrast|Adjusts the gamma contrast of an image.|U8, U16, U32, F32|NHWC, HWC|Both|
|Histogram|Calculates a histogram of values from a grayscale image.|U8|NHWC, HWC|Both|
|NonMaximumSuppression|Performs non-maximum suppression on batches of bounding boxes based on a score and IoU threshold.|S16, 4S16|NW, NWC|Both|
|Normalize|Normalizes an input tensor using a provided mean and standard deviation.|U8, S8, F32|NHWC, HWC|Both|
|Remap|Maps pixels in an image from one projection to another projection in a new image.|U8|NHWC, HWC|Both|
|Resize|Resizes an input tensor with interpolation.|U8, S8, F32|NHWC, HWC|Both|
|Rotate|Rotates (and optionally shifts) an input tensor by given angle in degrees counter-clockwise.|U8, S8, U16, S16, U32, S32, F32, F64|NHWC, HWC|Both|
|Threshold|Clamps values in an image to a global threshold value.|U8|NHWC, HWC|Both|
|WarpAffine|Performs an affine warp on an input tensor.|U8, S8, U16, S16, U32, S32, F32, F64|NHWC, HWC|Both|
|WarpPerspective|Performs a perspective warp on an input image.|U8, S8, U16, S16, U32, S32, F32, F64|NHWC, HWC|Both|
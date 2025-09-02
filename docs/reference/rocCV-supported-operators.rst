.. meta::
  :description: rocCV supported operators 
  :keywords: rocCV, ROCm, operators, support

*********************************************
rocCV supported operators
*********************************************

The rocCV is a collection of the following computer vision operators that are supported on both GPU and CPU:


.. csv-table::
    :header: "Operator", "Description", "Datatypes", "Layouts"
    
    "BilateralFilter", "Applies a bilateral filter.", "U8", "NHWC, HWC"
    "BndBox","Draws bounding boxes on the images in a tensor.","U8","NHWC, HWC"
    "Composite","Composites two input tensors using a provided alpha mask.","U8, S8, U32, S32, F32","NHWC, HWC"
    "CvtColor","Converts the color space of the images in a tensor.","U8","NHWC, HWC"
    "CopyMakeBorder","Generates a border using a specified border mode around the input tensor.","U8, S8, U32, S32, F32","NHWC"
    "CustomCrop","Crops a region of interest from an input tensor.","U8","NHWC, HWC"
    "GammaContrast","Adjusts the gamma contrast on images in a tensor.","U8","NHWC, HWC"
    "Histogram","Calculates a histogram of values from a grayscale image.","U8","NHWC, HWC"
    "NonMaximumSuppression","Performs non-maximum suppression on batches of bounding boxes based on a score and IoU threshold.","S16, 4S16","NW, NWC"
    "Normalize","Normalizes an input tensor using a provided mean and standard deviation.","U8, S8, F32","NHWC, HWC"
    "Remap","Maps pixels in an image from one projection to another projection in a new image.","U8","NHWC, HWC"
    "Resize","Resizes an input tensor with interpolation.","U8, S8, F32","NHWC, HWC"
    "Rotate","Rotates (and optionally shifts) an input tensor by given angle in degrees clockwise.","U8, S8, F32","NHWC, HWC"
    "Threshold","Clamps values in an image to a global threshold value.","U8","NHWC, HWC"
    "WarpAffine","Performs an affine warp on an input tensor.","U8, S8, F32","NHWC, HWC"
    "WarpPerspective","Performs a perspective warp on an input image.","U8, F32","NHWC, HWC"
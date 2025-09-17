from typing import ClassVar, overload

4S16: eDataType
BGR: eChannelType
BINARY: eThresholdType
BINARY_INV: eThresholdType
BOTH: eAxis
BT2020: eColorSpec
BT601: eColorSpec
BT709: eColorSpec
COLOR_BGR2GRAY: eColorConversionCode
COLOR_BGR2RGB: eColorConversionCode
COLOR_BGR2YUV: eColorConversionCode
COLOR_RGB2BGR: eColorConversionCode
COLOR_RGB2GRAY: eColorConversionCode
COLOR_RGB2YUV: eColorConversionCode
COLOR_YUV2BGR: eColorConversionCode
COLOR_YUV2RGB: eColorConversionCode
CONSTANT: eBorderType
CPU: eDeviceType
CUBIC: eInterpolationType
F32: eDataType
F64: eDataType
GPU: eDeviceType
Grayscale: eChannelType
HWC: eTensorLayout
LINEAR: eInterpolationType
N: eTensorLayout
NC: eTensorLayout
NCHW: eTensorLayout
NEAREST: eInterpolationType
NHWC: eTensorLayout
NW: eTensorLayout
NWC: eTensorLayout
REFLECT: eBorderType
REMAP_ABSOLUTE: eRemapType
REMAP_ABSOLUTE_NORMALIZED: eRemapType
REMAP_RELATIVE_NORMALIZED: eRemapType
REPLICATE: eBorderType
RGB: eChannelType
S16: eDataType
S32: eDataType
S8: eDataType
TOZERO: eThresholdType
TOZERO_INV: eThresholdType
TRUNC: eThresholdType
U16: eDataType
U32: eDataType
U8: eDataType
WRAP: eBorderType
X: eAxis
Y: eAxis
YUV: eChannelType
YVU: eChannelType

class BndBox:
    borderColor: ColorRGBA
    box: Box
    fillColor: ColorRGBA
    thickness: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.BndBox) -> None

        2. __init__(self: rocpycv.rocpycv.BndBox, box: rocpycv.rocpycv.Box, thickness: int, borderColor: rocpycv.rocpycv.ColorRGBA, fillColor: rocpycv.rocpycv.ColorRGBA) -> None
        """
    @overload
    def __init__(self, box: Box, thickness: int, borderColor: ColorRGBA, fillColor: ColorRGBA) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.BndBox) -> None

        2. __init__(self: rocpycv.rocpycv.BndBox, box: rocpycv.rocpycv.Box, thickness: int, borderColor: rocpycv.rocpycv.ColorRGBA, fillColor: rocpycv.rocpycv.ColorRGBA) -> None
        """

class BndBoxes:
    def __init__(self, bndboxes: list[list[BndBox]]) -> None:
        """__init__(self: rocpycv.rocpycv.BndBoxes, bndboxes: list[list[rocpycv.rocpycv.BndBox]]) -> None"""

class Box:
    height: int
    width: int
    x: int
    y: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.Box) -> None

        2. __init__(self: rocpycv.rocpycv.Box, x: int, y: int, width: int, height: int) -> None
        """
    @overload
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.Box) -> None

        2. __init__(self: rocpycv.rocpycv.Box, x: int, y: int, width: int, height: int) -> None
        """

class ColorRGBA:
    c0: int
    c1: int
    c2: int
    c3: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.ColorRGBA) -> None

        2. __init__(self: rocpycv.rocpycv.ColorRGBA, r: int, g: int, b: int, a: int) -> None
        """
    @overload
    def __init__(self, r: int, g: int, b: int, a: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.ColorRGBA) -> None

        2. __init__(self: rocpycv.rocpycv.ColorRGBA, r: int, g: int, b: int, a: int) -> None
        """

class Exception(Exception): ...

class NormalizeFlags:
    __members__: ClassVar[dict] = ...  # read-only
    SCALE_IS_STDDEV: ClassVar[NormalizeFlags] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.NormalizeFlags, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.NormalizeFlags) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.NormalizeFlags) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Size2D:
    h: int
    w: int
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.Size2D) -> None

        2. __init__(self: rocpycv.rocpycv.Size2D, w: int, h: int) -> None
        """
    @overload
    def __init__(self, w: int, h: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: rocpycv.rocpycv.Size2D) -> None

        2. __init__(self: rocpycv.rocpycv.Size2D, w: int, h: int) -> None
        """

class Stream:
    def __init__(self) -> None:
        """__init__(self: rocpycv.rocpycv.Stream) -> None

        Creates a HIP stream.
        """
    def synchronize(self) -> None:
        """synchronize(self: rocpycv.rocpycv.Stream) -> None

        Blocks until all worked queued on this stream is finished.
        """

class Tensor:
    def __init__(self, shape: list[int], layout: eTensorLayout, dtype: eDataType, device: eDeviceType = ...) -> None:
        """__init__(self: rocpycv.rocpycv.Tensor, shape: list[int], layout: rocpycv.rocpycv.eTensorLayout, dtype: rocpycv.rocpycv.eDataType, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None

        Constructs a tensor object.
        """
    def copy_to(self, device: eDeviceType) -> Tensor:
        """copy_to(self: rocpycv.rocpycv.Tensor, device: rocpycv.rocpycv.eDeviceType) -> rocpycv.rocpycv.Tensor

        Returns a deep copy of the tensor with data copied to a specified device type.
        """
    def device(self) -> eDeviceType:
        """device(self: rocpycv.rocpycv.Tensor) -> rocpycv.rocpycv.eDeviceType

        Returns the device this tensor is on.
        """
    def dtype(self) -> eDataType:
        """dtype(self: rocpycv.rocpycv.Tensor) -> rocpycv.rocpycv.eDataType

        Returns the data type of the tensor.
        """
    def layout(self) -> eTensorLayout:
        """layout(self: rocpycv.rocpycv.Tensor) -> rocpycv.rocpycv.eTensorLayout

        Returns the layout for this tensor.
        """
    def ndim(self) -> int:
        """ndim(self: rocpycv.rocpycv.Tensor) -> int

        Returns the number of dimensions of the tensor.
        """
    def reshape(self, new_shape: list[int], layout: eTensorLayout) -> Tensor:
        """reshape(self: rocpycv.rocpycv.Tensor, new_shape: list[int], layout: rocpycv.rocpycv.eTensorLayout) -> rocpycv.rocpycv.Tensor

        Creates a new tensor with the specified shape.
        """
    def shape(self) -> list[int]:
        """shape(self: rocpycv.rocpycv.Tensor) -> list[int]

        Returns a list representing the tensor shape.
        """
    def strides(self) -> list[int]:
        """strides(self: rocpycv.rocpycv.Tensor) -> list[int]

        Returns a list representing tensor strides.
        """
    def __dlpack__(self, stream: object = ...) -> capsule:
        """__dlpack__(self: rocpycv.rocpycv.Tensor, stream: object = None) -> capsule

        Creates a DLPack compatible tensor from this tensor.
        """
    def __dlpack_device__(self) -> tuple:
        """__dlpack_device__(self: rocpycv.rocpycv.Tensor) -> tuple

        Returns a tuple containing the DLPack device and device id for the tensor.
        """

class eAxis:
    __members__: ClassVar[dict] = ...  # read-only
    BOTH: ClassVar[eAxis] = ...
    X: ClassVar[eAxis] = ...
    Y: ClassVar[eAxis] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eAxis, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eAxis) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eAxis) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eBorderType:
    __members__: ClassVar[dict] = ...  # read-only
    CONSTANT: ClassVar[eBorderType] = ...
    REFLECT: ClassVar[eBorderType] = ...
    REPLICATE: ClassVar[eBorderType] = ...
    WRAP: ClassVar[eBorderType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eBorderType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eBorderType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eBorderType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eChannelType:
    __members__: ClassVar[dict] = ...  # read-only
    BGR: ClassVar[eChannelType] = ...
    Grayscale: ClassVar[eChannelType] = ...
    RGB: ClassVar[eChannelType] = ...
    YUV: ClassVar[eChannelType] = ...
    YVU: ClassVar[eChannelType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eChannelType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eChannelType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eChannelType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eColorConversionCode:
    __members__: ClassVar[dict] = ...  # read-only
    COLOR_BGR2GRAY: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2RGB: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2YUV: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2BGR: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2GRAY: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2YUV: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2BGR: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2RGB: ClassVar[eColorConversionCode] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eColorConversionCode, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eColorConversionCode) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eColorConversionCode) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eColorSpec:
    __members__: ClassVar[dict] = ...  # read-only
    BT2020: ClassVar[eColorSpec] = ...
    BT601: ClassVar[eColorSpec] = ...
    BT709: ClassVar[eColorSpec] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eColorSpec, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eColorSpec) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eColorSpec) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eDataType:
    __members__: ClassVar[dict] = ...  # read-only
    4S16: ClassVar[eDataType] = ...
    F32: ClassVar[eDataType] = ...
    F64: ClassVar[eDataType] = ...
    S16: ClassVar[eDataType] = ...
    S32: ClassVar[eDataType] = ...
    S8: ClassVar[eDataType] = ...
    U16: ClassVar[eDataType] = ...
    U32: ClassVar[eDataType] = ...
    U8: ClassVar[eDataType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eDataType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eDataType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eDataType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eDeviceType:
    __members__: ClassVar[dict] = ...  # read-only
    CPU: ClassVar[eDeviceType] = ...
    GPU: ClassVar[eDeviceType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eDeviceType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eDeviceType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eDeviceType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eInterpolationType:
    __members__: ClassVar[dict] = ...  # read-only
    CUBIC: ClassVar[eInterpolationType] = ...
    LINEAR: ClassVar[eInterpolationType] = ...
    NEAREST: ClassVar[eInterpolationType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eInterpolationType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eInterpolationType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eInterpolationType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eRemapType:
    __members__: ClassVar[dict] = ...  # read-only
    REMAP_ABSOLUTE: ClassVar[eRemapType] = ...
    REMAP_ABSOLUTE_NORMALIZED: ClassVar[eRemapType] = ...
    REMAP_RELATIVE_NORMALIZED: ClassVar[eRemapType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eRemapType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eRemapType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eRemapType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eTensorLayout:
    __members__: ClassVar[dict] = ...  # read-only
    HWC: ClassVar[eTensorLayout] = ...
    N: ClassVar[eTensorLayout] = ...
    NC: ClassVar[eTensorLayout] = ...
    NCHW: ClassVar[eTensorLayout] = ...
    NHWC: ClassVar[eTensorLayout] = ...
    NW: ClassVar[eTensorLayout] = ...
    NWC: ClassVar[eTensorLayout] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eTensorLayout, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eTensorLayout) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eTensorLayout) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class eThresholdType:
    __members__: ClassVar[dict] = ...  # read-only
    BINARY: ClassVar[eThresholdType] = ...
    BINARY_INV: ClassVar[eThresholdType] = ...
    TOZERO: ClassVar[eThresholdType] = ...
    TOZERO_INV: ClassVar[eThresholdType] = ...
    TRUNC: ClassVar[eThresholdType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eThresholdType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: rocpycv.rocpycv.eThresholdType) -> int"""
    def __int__(self) -> int:
        """__int__(self: rocpycv.rocpycv.eThresholdType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def bilateral_filter(src: Tensor, diameter: int, sigmaColor: float, sigmaSpace: float, borderMode: eBorderType, borderValue: list, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """bilateral_filter(src: rocpycv.rocpycv.Tensor, diameter: int, sigmaColor: float, sigmaSpace: float, borderMode: rocpycv.rocpycv.eBorderType, borderValue: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


        
                Executes the Bilateral Filter operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    diameter (int): bilateral filter diameter.
                    sigmaColor (float): Gaussian exponent for color difference, expected to be positive, if it isn't, will be set to 1.0
                    sigmaSpace (float): Gaussian exponent for position difference expected to be positive, if it isn't, will be set to 1.0
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def bilateral_filter_into(dst: Tensor, src: Tensor, diameter: int, sigmaColor: float, sigmaSpace: float, borderMode: eBorderType, borderValue: list, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """bilateral_filter_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, diameter: int, sigmaColor: float, sigmaSpace: float, borderMode: rocpycv.rocpycv.eBorderType, borderValue: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



            
                Executes the Bilateral Filter operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): The output tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    diameter (int): bilateral filter diameter.
                    sigmaColor (float): Gaussian exponent for color difference, expected to be positive, if it isn't, will be set to 1.0
                    sigmaSpace (float): Gaussian exponent for position difference expected to be positive, if it isn't, will be set to 1.0
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    None
           
    """
def bndbox(src: Tensor, bnd_boxes: BndBoxes, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """bndbox(src: rocpycv.rocpycv.Tensor, bnd_boxes: rocpycv.rocpycv.BndBoxes, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


            
                Executes the BndBox operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    bnd_boxes (rocpycv.BndBoxes): Bounding boxes to apply to input tensor.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.

            
    """
def bndbox_into(dst: Tensor, src: Tensor, bnd_boxes: BndBoxes, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """bndbox_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, bnd_boxes: rocpycv.rocpycv.BndBoxes, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



                Executes the BndBox operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.

                Args:
                    dst (rocpycv.Tensor): The output tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    bnd_boxes (rocpycv.BndBoxes): Bounding boxes to apply to input tensor.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
            
    """
def center_crop(src: Tensor, crop_size: tuple, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """center_crop(src: rocpycv.rocpycv.Tensor, crop_size: tuple, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Center Crop operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): Output tensor which image results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    crop_size (Tuple[int]): The crop rectangle width and height.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def center_crop_into(dst: Tensor, src: Tensor, crop_size: tuple, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """center_crop_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, crop_size: tuple, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Center Crop operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): Output tensor which image results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    crop_size (Tuple[int]): The crop rectangle width and height.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def composite(foreground: Tensor, background: Tensor, fgmask: Tensor, outchannels: int, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """composite(foreground: rocpycv.rocpycv.Tensor, background: rocpycv.rocpycv.Tensor, fgmask: rocpycv.rocpycv.Tensor, outchannels: int, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Composite operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    foreground (rocpycv.Tensor): Input foreground image.
                    background (rocpycv.Tensor): Input background image.
                    fgmask (rocpycv.Tensor): Grayscale alpha mask for compositing.
                    outchannels (int): Number of output channels for the output tensor. Must be 3 or 4. If 4, an alpha channel set to the max value will be added.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor with <outchannels> number of channels.
          
    """
def composite_into(dst: Tensor, foreground: Tensor, background: Tensor, fgmask: Tensor, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """composite_into(dst: rocpycv.rocpycv.Tensor, foreground: rocpycv.rocpycv.Tensor, background: rocpycv.rocpycv.Tensor, fgmask: rocpycv.rocpycv.Tensor, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Composite operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The output tensor with <outchannels> number of channels. Results will be written to this tensor.
                    foreground (rocpycv.Tensor): Input foreground image.
                    background (rocpycv.Tensor): Input background image.
                    fgmask (rocpycv.Tensor): Grayscale alpha mask for compositing.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def copymakeborder(src: Tensor, border_mode: eBorderType = ..., border_value: list = ..., top: int, bottom: int, left: int, right: int, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """copymakeborder(src: rocpycv.rocpycv.Tensor, border_mode: rocpycv.rocpycv.eBorderType = <eBorderType.CONSTANT: 0>, border_value: list = [0.0, 0.0, 0.0, 0.0], top: int, bottom: int, left: int, right: int, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the CopyMakeBorder operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input image tensor.
                    border_mode (rocpycv.eBorderType): Border type.
                    border_value (List[float]): Border values to use when using constant border type.
                    top (int): Top border height in pixels.
                    bottom (int): Bottom border height in pixels.
                    left (int): Left border width in pixels.
                    right (int): Right border width in pixels.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def copymakeborder_into(dst: Tensor, src: Tensor, border_mode: eBorderType = ..., border_value: list = ..., top: int, left: int, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """copymakeborder_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, border_mode: rocpycv.rocpycv.eBorderType = <eBorderType.CONSTANT: 0>, border_value: list = [0.0, 0.0, 0.0, 0.0], top: int, left: int, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the CopyMakeBorder operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The destination tensor.
                    src (rocpycv.Tensor): Input image tensor.
                    border_mode (rocpycv.eBorderType): Border type.
                    border_value (List[float]): Border values to use when using constant border type.
                    top (int): Top border height in pixels.
                    left (int): Left border width in pixels.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def custom_crop(src: Tensor, crop_rect: Box, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """custom_crop(src: rocpycv.rocpycv.Tensor, crop_rect: rocpycv.rocpycv.Box, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Custom Crop operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): Output tensor which image results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    crop_rect (rocpycv.Box): A Box defining how the image should be cropped.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def custom_crop_into(dst: Tensor, src: Tensor, crop_rect: Box, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """custom_crop_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, crop_rect: rocpycv.rocpycv.Box, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Custom Crop operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    crop_rect (rocpycv.Box): A Box defining how the image should be cropped.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def cvtcolor(src: Tensor, conversion_code: eColorConversionCode, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """cvtcolor(src: rocpycv.rocpycv.Tensor, conversion_code: rocpycv.rocpycv.eColorConversionCode, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


    
                Executes the Color Convert operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    conversion_code (eColorConversionCode): Conversion code specifying the formats being converted (ex. COLOR_RGB2YUV)
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def cvtcolor_into(dst: Tensor, src: Tensor, conversion_code: eColorConversionCode, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """cvtcolor_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, conversion_code: rocpycv.rocpycv.eColorConversionCode, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



                Executes the Color Convert operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): Output tensor for storing modified image data.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    conversion_code (eColorConversionCode): Conversion code specifying the formats being converted (ex. COLOR_RGB2YUV)
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    None
          
    """
def flip(src: Tensor, flip_code: int, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """flip(src: rocpycv.rocpycv.Tensor, flip_code: int, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Flip operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    flip_code (int): A flip code representing how images in the batch should be flipped. 
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def flip_into(dst: Tensor, src: Tensor, flip_code: int, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """flip_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, flip_code: int, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Flip operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The destination tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    flip_code (int): A flip code representing how images in the batch should be flipped. 
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on. 0 flips along the x-axis, positive integer flips along the y-axis, and negative integers flip along both axis.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def from_dlpack(buffer: object, layout: eTensorLayout) -> Tensor:
    """from_dlpack(buffer: object, layout: rocpycv.rocpycv.eTensorLayout) -> rocpycv.rocpycv.Tensor

    Wraps a DLPack supported tensor in a rocpycv tensor.
    """
def gamma_contrast(src: Tensor, gamma: float, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """gamma_contrast(src: rocpycv.rocpycv.Tensor, gamma: float, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


            
                Executes the Gamma Contrast operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    gamma (float): Gamma correction value to apply to the images.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    rocpycv.Tensor: The output tensor.
        
    """
def gamma_contrast_into(dst: Tensor, src: Tensor, gamma: float, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """gamma_contrast_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, gamma: float, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


            
                Executes the Gamma Contrast operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): The output tensor with gamma correction applied.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    gamma (float): Gamma correction value to apply to the images.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    None
        
    """
def histogram(src: Tensor, mask: Tensor | None, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """histogram(src: rocpycv.rocpycv.Tensor, mask: Optional[rocpycv.rocpycv.Tensor], stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor



                Executes the Histogram operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    mask (rocpycv.Tensor): (Optional) Mask tensor with shape equal to the input tensor shape and any value not equal 0 will be counted in histogram.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    rocpycv.Tensor: Output tensor with width of 256 and a height equal to the batch size of input (1 if HWC input).

    
    """
def histogram_into(dst: Tensor, src: Tensor, mask: Tensor | None, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """histogram_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, mask: Optional[rocpycv.rocpycv.Tensor], stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



                Executes the Histogram operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): Output tensor with width of 256 and a height equal to the batch size of input (1 if HWC input).
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    mask (rocpycv.Tensor): (Optional) Mask tensor with shape equal to the input tensor shape and any value not equal 0 will be counted in histogram.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    None

    
    """
def nms(src: Tensor, scores: Tensor, score_threshold: float = ..., iou_threshold: float = ..., stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """nms(src: rocpycv.rocpycv.Tensor, scores: rocpycv.rocpycv.Tensor, score_threshold: float = 1.1920928955078125e-07, iou_threshold: float = 1.0, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Non-maximum Suppression operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): An input tensor of size [i, j, 4] containing bounding boxes with the following structure (x, y, width, height).
                    scores (rocpycv.Tensor): A size [i, j] tensor containing confidence scores for each box j in batch i.
                    score_threshold (float): Minimum score an input bounding box proposal needs to be kept.
                    iou_threshold (float): The IoU threshold to filter overlapping boxes. Defaults to 1.0.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor of shape [i, j], containing 1 (kept) or 0 (suppressed) for each bounding box (j) per batch (i). Results will be written to this tensor.
          
    """
def nms_into(dst: Tensor, src: Tensor, scores: Tensor, score_threshold: float = ..., iou_threshold: float = ..., stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """nms_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, scores: rocpycv.rocpycv.Tensor, score_threshold: float = 1.1920928955078125e-07, iou_threshold: float = 1.0, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Non-maximum Suppression operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The output tensor of shape [i, j], containing 1 (kept) or 0 (suppressed) for each bounding box (j) per batch (i). Results will be written to this tensor.
                    src (rocpycv.Tensor): An input tensor of size [i, j, 4] containing bounding boxes with the following structure (x, y, width, height).
                    scores (rocpycv.Tensor): A size [i, j] tensor containing confidence scores for each box j in batch i.
                    score_threshold (float): Minimum score an input bounding box proposal needs to be kept.
                    iou_threshold (float): The IoU threshold to filter overlapping boxes. Defaults to 1.0.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def normalize(src: Tensor, base: Tensor, scale: Tensor, flags: int | None = ..., globalscale: float = ..., globalshift: float = ..., epsilon: float = ..., stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """normalize(src: rocpycv.rocpycv.Tensor, base: rocpycv.rocpycv.Tensor, scale: rocpycv.rocpycv.Tensor, flags: Optional[int] = None, globalscale: float = 1.0, globalshift: float = 0.0, epsilon: float = 0.0, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Normalize operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    base (rocpycv.Tensor): Tensor for base values.
                    scale (rocpycv.Tensor): Tensor for scale values.
                    flags (int): Flags for the Normalize operation. Use NormalizeFlags.SCALE_IS_STDDEV to interpret the scale tensor as standard deviation instead.
                    globalscale (float): Scale factor applied after the mean is subtracted and the standard deviation is divided. Defaults to 1.
                    globalshift (float): The values of the final image will be shifted by this amount after scaling. Defaults to 0.
                    epsilon (float): Epsilon value for numerical stability. Defaults to 0.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def normalize_into(dst: Tensor, src: Tensor, base: Tensor, scale: Tensor, flags: int | None = ..., globalscale: float = ..., globalshift: float = ..., epsilon: float = ..., stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """normalize_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, base: rocpycv.rocpycv.Tensor, scale: rocpycv.rocpycv.Tensor, flags: Optional[int] = None, globalscale: float = 1.0, globalshift: float = 0.0, epsilon: float = 0.0, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


            
                  Executes the Normalize operation on the given HIP stream.
  
                  See also:
                      Refer to the rocCV C++ API reference for more information on this operation.
          
                  Args:
                      dst (rocpycv.Tensor): The output tensor which results are written to.
                      src (rocpycv.Tensor): Input tensor containing one or more images.
                      base (rocpycv.Tensor): Tensor for base values.
                      scale (rocpycv.Tensor): Tensor for scale values.
                      flags (int): Flags for the Normalize operation. Use NormalizeFlags.SCALE_IS_STDDEV to interpret the scale tensor as standard deviation instead.
                      globalscale (float): Scale factor applied after the mean is subtracted and the standard deviation is divided. Defaults to 1.
                      globalshift (float): The values of the final image will be shifted by this amount after scaling. Defaults to 0.
                      epsilon (float): Epsilon value for numerical stability. Defaults to 0.
                      stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                      device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
              
                  Returns:
                      None
            
    """
def remap(src: Tensor, map: Tensor, in_interpolation: eInterpolationType, map_interpolation: eInterpolationType, map_value_type: eRemapType, align_corners: bool, border_type: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """remap(src: rocpycv.rocpycv.Tensor, map: rocpycv.rocpycv.Tensor, in_interpolation: rocpycv.rocpycv.eInterpolationType, map_interpolation: rocpycv.rocpycv.eInterpolationType, map_value_type: rocpycv.rocpycv.eRemapType, align_corners: bool, border_type: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Remap operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    map (rocpycv.Tensor): Map tensor containing absolute or relative positions for how to remap the pixels of the input tensor to the output tensor
                    in_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting values from the input tensor.
                    map_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting indices from the map tensor.
                    map_value_type (rocpycv.eRemapType): Determines how the values in the map are interpreted.
                    align_corners (bool): Set to true if corner values are aligned to center points of corner pixels and set to false if they are aligned by the corner points of the corner pixels.
                    border_type (rocpycv.eBorderType): A border type to identify the pixel extrapolation method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def remap_into(dst: Tensor, src: Tensor, map: Tensor, in_interpolation: eInterpolationType, map_interpolation: eInterpolationType, map_value_type: eRemapType, align_corners: bool, border_type: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """remap_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, map: rocpycv.rocpycv.Tensor, in_interpolation: rocpycv.rocpycv.eInterpolationType, map_interpolation: rocpycv.rocpycv.eInterpolationType, map_value_type: rocpycv.rocpycv.eRemapType, align_corners: bool, border_type: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Remap operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The output tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    map (rocpycv.Tensor): Map tensor containing absolute or relative positions for how to remap the pixels of the input tensor to the output tensor
                    in_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting values from the input tensor.
                    map_interpolation (rocpycv.eInterpolationType): Interpolation type to be used when getting indices from the map tensor.
                    map_value_type (rocpycv.eRemapType): Determines how the values in the map are interpreted.
                    align_corners (bool): Set to true if corner values are aligned to center points of corner pixels and set to false if they are aligned by the corner points of the corner pixels.
                    border_type (rocpycv.eBorderType): A border type to identify the pixel extrapolation method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def resize(src: Tensor, shape: tuple, interp: eInterpolationType, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """resize(src: rocpycv.rocpycv.Tensor, shape: tuple, interp: rocpycv.rocpycv.eInterpolationType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Resize operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    shape (Tuple[int]): Shape of the output tensor.
                    interp (rocpycv.eInterpolationType): Interpolation type used for transform.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def resize_into(dst: Tensor, src: Tensor, interp: eInterpolationType, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """resize_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, interp: rocpycv.rocpycv.eInterpolationType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Resize operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): Output tensor which stores the result of the operation.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    interp (rocpycv.eInterpolationType): Interpolation type used for transform.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def rotate(src: Tensor, angle_deg: float, shift: tuple, interpolation: eInterpolationType, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """rotate(src: rocpycv.rocpycv.Tensor, angle_deg: float, shift: tuple, interpolation: rocpycv.rocpycv.eInterpolationType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Rotate operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    angle_deg (float): The angle in degrees to rotate the images by.
                    shift (Tuple[float]): x and y coordinates to shift the rotated image by.
                    interpolation (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def rotate_into(dst: Tensor, src: Tensor, angle_deg: float, shift: tuple, interpolation: eInterpolationType, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """rotate_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, angle_deg: float, shift: tuple, interpolation: rocpycv.rocpycv.eInterpolationType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Rotate operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The output tensor to which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    angle_deg (float): The angle in degrees to rotate the images by.
                    shift (Tuple[float]): x and y coordinates to shift the rotated image by.
                    interpolation (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def threshold(src: Tensor, thresh: Tensor, maxVal: Tensor, maxBatchSize: int, threshType: eThresholdType, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """threshold(src: rocpycv.rocpycv.Tensor, thresh: rocpycv.rocpycv.Tensor, maxVal: rocpycv.rocpycv.Tensor, maxBatchSize: int, threshType: rocpycv.rocpycv.eThresholdType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


            
                Executes the Thresholding operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    thresh (rocpycv.Tensor): thresh an array of size maxBatch that gives the threshold value of each image.
                    maxVal (rocpycv.Tensor): maxval an array of size maxBatch that gives the maxval value of each image, used with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV thresholding types.
                    maxBatchSize (uint32_t): The maximum batch size.
                    threshType (eThresholdType): Threshold type
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

    
    """
def threshold_into(dst: Tensor, src: Tensor, thresh: Tensor, maxVal: Tensor, maxBatchSize: int, threshType: eThresholdType, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """threshold_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, thresh: rocpycv.rocpycv.Tensor, maxVal: rocpycv.rocpycv.Tensor, maxBatchSize: int, threshType: rocpycv.rocpycv.eThresholdType, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



                Executes the Thresholding operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): The output tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    thresh (rocpycv.Tensor): thresh an array of size maxBatch that gives the threshold value of each image.
                    maxVal (rocpycv.Tensor): maxval an array of size maxBatch that gives the maxval value of each image, used with the NVCV_THRESH_BINARY and NVCV_THRESH_BINARY_INV thresholding types.
                    maxBatchSize (uint32_t): The maximum batch size.
                    threshType (eThresholdType): Threshold type
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

    
    """
def warp_affine(src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """warp_affine(src: rocpycv.rocpycv.Tensor, xform: list, inverted: bool, interp: rocpycv.rocpycv.eInterpolationType, border_mode: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Warp Affine operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    xform (List[float]): The input affine transformation matrix in row-major order. Must have 6 elements.
                    inverted (bool): Marks the transformation matrix as inverted or not.
                    interp (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def warp_affine_into(dst: Tensor, src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """warp_affine_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, xform: list, inverted: bool, interp: rocpycv.rocpycv.eInterpolationType, border_mode: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Warp Affine operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): Output tensor to which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    xform (List[float]): The input affine transformation matrix in row-major order. Must have 6 elements.
                    inverted (bool): Marks the transformation matrix as inverted or not.
                    interp (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
def warp_perspective(src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """warp_perspective(src: rocpycv.rocpycv.Tensor, xform: list, inverted: bool, interp: rocpycv.rocpycv.eInterpolationType, border_mode: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


          
                Executes the Warp Perspective operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    xform (List[float]): A transformation matrix representing the perspective transformation.
                    inverted (bool): Marks the transformation matrix as inverted or not.
                    interp (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def warp_perspective_into(dst: Tensor, src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """warp_perspective_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, xform: list, inverted: bool, interp: rocpycv.rocpycv.eInterpolationType, border_mode: rocpycv.rocpycv.eBorderType, border_value: list, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None


          
                Executes the Warp Perspective operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
        
                Args:
                    dst (rocpycv.Tensor): The output tensor which results are written to.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    xform (List[float]): A transformation matrix representing the perspective transformation.
                    inverted (bool): Marks the transformation matrix as inverted or not.
                    interp (rocpycv.eInterpolationType): The interpolation method to use for the output images.
                    border_mode (rocpycv.eBorderType): The border type to identify the pixel extrapolation method.
                    border_value (List[float]): The color value to use when a constant border is selected.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.
            
                Returns:
                    None
          
    """
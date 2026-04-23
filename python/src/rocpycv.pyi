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
COLOR_BGR2YUV_NV12: eColorConversionCode
COLOR_BGR2YUV_NV21: eColorConversionCode
COLOR_RGB2BGR: eColorConversionCode
COLOR_RGB2GRAY: eColorConversionCode
COLOR_RGB2YUV: eColorConversionCode
COLOR_RGB2YUV_NV12: eColorConversionCode
COLOR_RGB2YUV_NV21: eColorConversionCode
COLOR_YUV2BGR: eColorConversionCode
COLOR_YUV2BGR_NV12: eColorConversionCode
COLOR_YUV2BGR_NV21: eColorConversionCode
COLOR_YUV2RGB: eColorConversionCode
COLOR_YUV2RGB_NV12: eColorConversionCode
COLOR_YUV2RGB_NV21: eColorConversionCode
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
REFLECT101: eBorderType
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
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, box: Box, thickness: typing.SupportsInt, borderColor: ColorRGBA, fillColor: ColorRGBA) -> None:
        ...
    @property
    def thickness(self) -> int:
        ...
    @thickness.setter
    def thickness(self, arg0: typing.SupportsInt) -> None:
        ...
class BndBoxes:
    def __init__(self, bndboxes: collections.abc.Sequence[collections.abc.Sequence[BndBox]]) -> None:
        ...
class Box:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsInt, y: typing.SupportsInt, width: typing.SupportsInt, height: typing.SupportsInt) -> None:
        ...
    @property
    def height(self) -> int:
        ...
    @height.setter
    def height(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def width(self) -> int:
        ...
    @width.setter
    def width(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def x(self) -> int:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def y(self) -> int:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsInt) -> None:
        ...
class ColorRGBA:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, r: typing.SupportsInt, g: typing.SupportsInt, b: typing.SupportsInt, a: typing.SupportsInt) -> None:
        ...
    @property
    def c0(self) -> int:
        ...
    @c0.setter
    def c0(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def c1(self) -> int:
        ...
    @c1.setter
    def c1(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def c2(self) -> int:
        ...
    @c2.setter
    def c2(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def c3(self) -> int:
        ...
    @c3.setter
    def c3(self, arg0: typing.SupportsInt) -> None:
        ...
class Exception(Exception):
    pass
class NormalizeFlags:
    """
    Members:
    
      SCALE_IS_STDDEV
    """
    SCALE_IS_STDDEV: typing.ClassVar[NormalizeFlags]  # value = <NormalizeFlags.SCALE_IS_STDDEV: 1>
    __members__: typing.ClassVar[dict[str, NormalizeFlags]]  # value = {'SCALE_IS_STDDEV': <NormalizeFlags.SCALE_IS_STDDEV: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Size2D:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, w: typing.SupportsInt, h: typing.SupportsInt) -> None:
        ...
    @property
    def h(self) -> int:
        ...
    @h.setter
    def h(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def w(self) -> int:
        ...
    @w.setter
    def w(self, arg0: typing.SupportsInt) -> None:
        ...
class Stream:
    """
    Python wrapper for HIP streams.
    """
    def __init__(self) -> None:
        """
        Creates a HIP stream.
        """
    def synchronize(self) -> None:
        """
        Blocks until all worked queued on this stream is finished.
        """
class Tensor:
    def __dlpack__(self, stream: typing.Any = None) -> typing_extensions.CapsuleType:
        """
        Creates a DLPack compatible tensor from this tensor.
        """
    def __dlpack_device__(self) -> tuple:
        """
        Returns a tuple containing the DLPack device and device id for the tensor.
        """
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt], layout: eTensorLayout, dtype: eDataType, device: eDeviceType = ...) -> None:
        """
        Constructs a tensor object.
        """
    def copy_to(self, device: eDeviceType) -> Tensor:
        """
        Returns a deep copy of the tensor with data copied to a specified device type.
        """
    def device(self) -> eDeviceType:
        """
        Returns the device this tensor is on.
        """
    def dtype(self) -> eDataType:
        """
        Returns the data type of the tensor.
        """
    def layout(self) -> eTensorLayout:
        """
        Returns the layout for this tensor.
        """
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the tensor.
        """
    def reshape(self, new_shape: collections.abc.Sequence[typing.SupportsInt], layout: eTensorLayout) -> Tensor:
        """
        Creates a new tensor with the specified shape.
        """
    def shape(self) -> list[int]:
        """
        Returns a list representing the tensor shape.
        """
    def strides(self) -> list[int]:
        """
        Returns a list representing tensor strides.
        """
class eAxis:
    """
    Members:
    
      X
    
      Y
    
      BOTH
    """
    BOTH: typing.ClassVar[eAxis]  # value = <eAxis.BOTH: -1>
    X: typing.ClassVar[eAxis]  # value = <eAxis.X: 0>
    Y: typing.ClassVar[eAxis]  # value = <eAxis.Y: 1>
    __members__: typing.ClassVar[dict[str, eAxis]]  # value = {'X': <eAxis.X: 0>, 'Y': <eAxis.Y: 1>, 'BOTH': <eAxis.BOTH: -1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eBorderType:
    """
    Members:
    
      CONSTANT
    
      REPLICATE
    
      REFLECT
    
      REFLECT101
    
      WRAP
    """
    CONSTANT: typing.ClassVar[eBorderType]  # value = <eBorderType.CONSTANT: 0>
    REFLECT: typing.ClassVar[eBorderType]  # value = <eBorderType.REFLECT: 2>
    REFLECT101: typing.ClassVar[eBorderType]  # value = <eBorderType.REFLECT101: 3>
    REPLICATE: typing.ClassVar[eBorderType]  # value = <eBorderType.REPLICATE: 1>
    WRAP: typing.ClassVar[eBorderType]  # value = <eBorderType.WRAP: 4>
    __members__: typing.ClassVar[dict[str, eBorderType]]  # value = {'CONSTANT': <eBorderType.CONSTANT: 0>, 'REPLICATE': <eBorderType.REPLICATE: 1>, 'REFLECT': <eBorderType.REFLECT: 2>, 'REFLECT101': <eBorderType.REFLECT101: 3>, 'WRAP': <eBorderType.WRAP: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eChannelType:
    """
    Members:
    
      RGB
    
      BGR
    
      YUV
    
      YVU
    
      Grayscale
    """
    BGR: typing.ClassVar[eChannelType]  # value = <eChannelType.BGR: 2>
    Grayscale: typing.ClassVar[eChannelType]  # value = <eChannelType.Grayscale: 16>
    RGB: typing.ClassVar[eChannelType]  # value = <eChannelType.RGB: 1>
    YUV: typing.ClassVar[eChannelType]  # value = <eChannelType.YUV: 4>
    YVU: typing.ClassVar[eChannelType]  # value = <eChannelType.YVU: 8>
    __members__: typing.ClassVar[dict[str, eChannelType]]  # value = {'RGB': <eChannelType.RGB: 1>, 'BGR': <eChannelType.BGR: 2>, 'YUV': <eChannelType.YUV: 4>, 'YVU': <eChannelType.YVU: 8>, 'Grayscale': <eChannelType.Grayscale: 16>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eColorConversionCode:
    __members__: ClassVar[dict] = ...  # read-only
    COLOR_BGR2GRAY: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2RGB: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2YUV: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2YUV_NV12: ClassVar[eColorConversionCode] = ...
    COLOR_BGR2YUV_NV21: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2BGR: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2GRAY: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2YUV: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2YUV_NV12: ClassVar[eColorConversionCode] = ...
    COLOR_RGB2YUV_NV21: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2BGR: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2BGR_NV12: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2BGR_NV21: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2RGB: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2RGB_NV12: ClassVar[eColorConversionCode] = ...
    COLOR_YUV2RGB_NV21: ClassVar[eColorConversionCode] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: rocpycv.rocpycv.eColorConversionCode, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eColorSpec:
    """
    Members:
    
      BT601
    
      BT709
    
      BT2020
    """
    BT2020: typing.ClassVar[eColorSpec]  # value = <eColorSpec.BT2020: 2>
    BT601: typing.ClassVar[eColorSpec]  # value = <eColorSpec.BT601: 0>
    BT709: typing.ClassVar[eColorSpec]  # value = <eColorSpec.BT709: 1>
    __members__: typing.ClassVar[dict[str, eColorSpec]]  # value = {'BT601': <eColorSpec.BT601: 0>, 'BT709': <eColorSpec.BT709: 1>, 'BT2020': <eColorSpec.BT2020: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eDataType:
    """
    Members:
    
      U8
    
      S8
    
      U16
    
      S16
    
      U32
    
      S32
    
      F32
    
      F64
    
      4S16
    """
    F32: typing.ClassVar[eDataType]  # value = <eDataType.F32: 6>
    F64: typing.ClassVar[eDataType]  # value = <eDataType.F64: 7>
    S16: typing.ClassVar[eDataType]  # value = <eDataType.S16: 3>
    S32: typing.ClassVar[eDataType]  # value = <eDataType.S32: 5>
    S8: typing.ClassVar[eDataType]  # value = <eDataType.S8: 1>
    U16: typing.ClassVar[eDataType]  # value = <eDataType.U16: 2>
    U32: typing.ClassVar[eDataType]  # value = <eDataType.U32: 4>
    U8: typing.ClassVar[eDataType]  # value = <eDataType.U8: 0>
    __members__: typing.ClassVar[dict[str, eDataType]]  # value = {'U8': <eDataType.U8: 0>, 'S8': <eDataType.S8: 1>, 'U16': <eDataType.U16: 2>, 'S16': <eDataType.S16: 3>, 'U32': <eDataType.U32: 4>, 'S32': <eDataType.S32: 5>, 'F32': <eDataType.F32: 6>, 'F64': <eDataType.F64: 7>, '4S16': <eDataType.4S16: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eDeviceType:
    """
    Members:
    
      GPU
    
      CPU
    """
    CPU: typing.ClassVar[eDeviceType]  # value = <eDeviceType.CPU: 1>
    GPU: typing.ClassVar[eDeviceType]  # value = <eDeviceType.GPU: 0>
    __members__: typing.ClassVar[dict[str, eDeviceType]]  # value = {'GPU': <eDeviceType.GPU: 0>, 'CPU': <eDeviceType.CPU: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eInterpolationType:
    """
    Members:
    
      NEAREST
    
      LINEAR
    
      CUBIC
    """
    CUBIC: typing.ClassVar[eInterpolationType]  # value = <eInterpolationType.CUBIC: 2>
    LINEAR: typing.ClassVar[eInterpolationType]  # value = <eInterpolationType.LINEAR: 1>
    NEAREST: typing.ClassVar[eInterpolationType]  # value = <eInterpolationType.NEAREST: 0>
    __members__: typing.ClassVar[dict[str, eInterpolationType]]  # value = {'NEAREST': <eInterpolationType.NEAREST: 0>, 'LINEAR': <eInterpolationType.LINEAR: 1>, 'CUBIC': <eInterpolationType.CUBIC: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eRemapType:
    """
    Members:
    
      REMAP_ABSOLUTE
    
      REMAP_ABSOLUTE_NORMALIZED
    
      REMAP_RELATIVE_NORMALIZED
    """
    REMAP_ABSOLUTE: typing.ClassVar[eRemapType]  # value = <eRemapType.REMAP_ABSOLUTE: 0>
    REMAP_ABSOLUTE_NORMALIZED: typing.ClassVar[eRemapType]  # value = <eRemapType.REMAP_ABSOLUTE_NORMALIZED: 1>
    REMAP_RELATIVE_NORMALIZED: typing.ClassVar[eRemapType]  # value = <eRemapType.REMAP_RELATIVE_NORMALIZED: 2>
    __members__: typing.ClassVar[dict[str, eRemapType]]  # value = {'REMAP_ABSOLUTE': <eRemapType.REMAP_ABSOLUTE: 0>, 'REMAP_ABSOLUTE_NORMALIZED': <eRemapType.REMAP_ABSOLUTE_NORMALIZED: 1>, 'REMAP_RELATIVE_NORMALIZED': <eRemapType.REMAP_RELATIVE_NORMALIZED: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eTensorLayout:
    """
    Members:
    
      NHWC
    
      HWC
    
      NC
    
      NW
    
      N
    
      NCHW
    
      NWC
    """
    HWC: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.HWC: 1>
    N: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.N: 4>
    NC: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.NC: 2>
    NCHW: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.NCHW: 8>
    NHWC: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.NHWC: 0>
    NW: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.NW: 3>
    NWC: typing.ClassVar[eTensorLayout]  # value = <eTensorLayout.NWC: 9>
    __members__: typing.ClassVar[dict[str, eTensorLayout]]  # value = {'NHWC': <eTensorLayout.NHWC: 0>, 'HWC': <eTensorLayout.HWC: 1>, 'NC': <eTensorLayout.NC: 2>, 'NW': <eTensorLayout.NW: 3>, 'N': <eTensorLayout.N: 4>, 'NCHW': <eTensorLayout.NCHW: 8>, 'NWC': <eTensorLayout.NWC: 9>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class eThresholdType:
    """
    Members:
    
      BINARY
    
      BINARY_INV
    
      TRUNC
    
      TOZERO
    
      TOZERO_INV
    """
    BINARY: typing.ClassVar[eThresholdType]  # value = <eThresholdType.BINARY: 1>
    BINARY_INV: typing.ClassVar[eThresholdType]  # value = <eThresholdType.BINARY_INV: 2>
    TOZERO: typing.ClassVar[eThresholdType]  # value = <eThresholdType.TOZERO: 8>
    TOZERO_INV: typing.ClassVar[eThresholdType]  # value = <eThresholdType.TOZERO_INV: 16>
    TRUNC: typing.ClassVar[eThresholdType]  # value = <eThresholdType.TRUNC: 4>
    __members__: typing.ClassVar[dict[str, eThresholdType]]  # value = {'BINARY': <eThresholdType.BINARY: 1>, 'BINARY_INV': <eThresholdType.BINARY_INV: 2>, 'TRUNC': <eThresholdType.TRUNC: 4>, 'TOZERO': <eThresholdType.TOZERO: 8>, 'TOZERO_INV': <eThresholdType.TOZERO_INV: 16>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def bilateral_filter(src: Tensor, diameter: typing.SupportsInt, sigmaColor: typing.SupportsFloat, sigmaSpace: typing.SupportsFloat, borderMode: eBorderType, borderValue: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def bilateral_filter_into(dst: Tensor, src: Tensor, diameter: typing.SupportsInt, sigmaColor: typing.SupportsFloat, sigmaSpace: typing.SupportsFloat, borderMode: eBorderType, borderValue: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def bndbox(src: Tensor, bnd_boxes: BndBoxes, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def bndbox_into(dst: Tensor, src: Tensor, bnd_boxes: BndBoxes, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def center_crop(src: Tensor, crop_size: tuple, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def center_crop_into(dst: Tensor, src: Tensor, crop_size: tuple, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def composite(foreground: Tensor, background: Tensor, fgmask: Tensor, outchannels: typing.SupportsInt, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def composite_into(dst: Tensor, foreground: Tensor, background: Tensor, fgmask: Tensor, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def copymakeborder(src: Tensor, border_mode: eBorderType = ..., border_value: list = [0.0, 0.0, 0.0, 0.0], top: typing.SupportsInt, bottom: typing.SupportsInt, left: typing.SupportsInt, right: typing.SupportsInt, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def copymakeborder_into(dst: Tensor, src: Tensor, border_mode: eBorderType = ..., border_value: list = [0.0, 0.0, 0.0, 0.0], top: typing.SupportsInt, left: typing.SupportsInt, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def custom_crop(src: Tensor, crop_rect: Box, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def custom_crop_into(dst: Tensor, src: Tensor, crop_rect: Box, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def cvtcolor(src: Tensor, conversion_code: eColorConversionCode, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def cvtcolor_into(dst: Tensor, src: Tensor, conversion_code: eColorConversionCode, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def advcvtcolor(src: Tensor, conversion_code: eColorConversionCode, color_spec: eColorSpec, stream: Stream | None = ..., device: eDeviceType = ...) -> Tensor:
    """advcvtcolor(src: rocpycv.rocpycv.Tensor, conversion_code: rocpycv.rocpycv.eColorConversionCode, color_spec: rocpycv.rocpycv.eColorSpec, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> rocpycv.rocpycv.Tensor


    
                Executes the Advanced Color Convert operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    conversion_code (eColorConversionCode): Conversion code specifying the formats being converted.
                    color_spec (eColorSpec): Color specification selecting BT601/BT709/BT2020 conversion matrices.
                    stream (rocpycv.Stream, optional): HIP stream to run this operation on.
                    device (rocpycv.Device, optional): The device to run this operation on. Defaults to GPU.

                Returns:
                    rocpycv.Tensor: The output tensor.
          
    """
def advcvtcolor_into(dst: Tensor, src: Tensor, conversion_code: eColorConversionCode, color_spec: eColorSpec, stream: Stream | None = ..., device: eDeviceType = ...) -> None:
    """advcvtcolor_into(dst: rocpycv.rocpycv.Tensor, src: rocpycv.rocpycv.Tensor, conversion_code: rocpycv.rocpycv.eColorConversionCode, color_spec: rocpycv.rocpycv.eColorSpec, stream: Optional[rocpycv.rocpycv.Stream] = None, device: rocpycv.rocpycv.eDeviceType = <eDeviceType.GPU: 0>) -> None



                Executes the Advanced Color Convert operation on the given HIP stream.

                See also:
                    Refer to the rocCV C++ API reference for more information on this operation.
            
                Args:
                    dst (rocpycv.Tensor): Output tensor for storing modified image data.
                    src (rocpycv.Tensor): Input tensor containing one or more images.
                    conversion_code (eColorConversionCode): Conversion code specifying the formats being converted.
                    color_spec (eColorSpec): Color specification selecting BT601/BT709/BT2020 conversion matrices.
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
def flip_into(dst: Tensor, src: Tensor, flip_code: typing.SupportsInt, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def from_dlpack(buffer: typing.Any, layout: eTensorLayout) -> Tensor:
    """
    Wraps a DLPack supported tensor in a rocpycv tensor.
    """
def gamma_contrast(src: Tensor, gamma: typing.SupportsFloat, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def gamma_contrast_into(dst: Tensor, src: Tensor, gamma: typing.SupportsFloat, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def histogram(src: Tensor, mask: rocpycv.Tensor | None, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def histogram_into(dst: Tensor, src: Tensor, mask: rocpycv.Tensor | None, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def nms(src: Tensor, scores: Tensor, score_threshold: typing.SupportsFloat = 1.1920928955078125e-07, iou_threshold: typing.SupportsFloat = 1.0, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def nms_into(dst: Tensor, src: Tensor, scores: Tensor, score_threshold: typing.SupportsFloat = 1.1920928955078125e-07, iou_threshold: typing.SupportsFloat = 1.0, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def normalize(src: Tensor, base: Tensor, scale: Tensor, flags: typing.SupportsInt | None = None, globalscale: typing.SupportsFloat = 1.0, globalshift: typing.SupportsFloat = 0.0, epsilon: typing.SupportsFloat = 0.0, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def normalize_into(dst: Tensor, src: Tensor, base: Tensor, scale: Tensor, flags: typing.SupportsInt | None = None, globalscale: typing.SupportsFloat = 1.0, globalshift: typing.SupportsFloat = 0.0, epsilon: typing.SupportsFloat = 0.0, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def remap(src: Tensor, map: Tensor, in_interpolation: eInterpolationType, map_interpolation: eInterpolationType, map_value_type: eRemapType, align_corners: bool, border_type: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def remap_into(dst: Tensor, src: Tensor, map: Tensor, in_interpolation: eInterpolationType, map_interpolation: eInterpolationType, map_value_type: eRemapType, align_corners: bool, border_type: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def resize(src: Tensor, shape: tuple, interp: eInterpolationType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def resize_into(dst: Tensor, src: Tensor, interp: eInterpolationType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def rotate(src: Tensor, angle_deg: typing.SupportsFloat, shift: tuple, interpolation: eInterpolationType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def rotate_into(dst: Tensor, src: Tensor, angle_deg: typing.SupportsFloat, shift: tuple, interpolation: eInterpolationType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def threshold(src: Tensor, thresh: Tensor, maxVal: Tensor, maxBatchSize: typing.SupportsInt, threshType: eThresholdType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def threshold_into(dst: Tensor, src: Tensor, thresh: Tensor, maxVal: Tensor, maxBatchSize: typing.SupportsInt, threshType: eThresholdType, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def warp_affine(src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def warp_affine_into(dst: Tensor, src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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
def warp_perspective(src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> Tensor:
    """
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
def warp_perspective_into(dst: Tensor, src: Tensor, xform: list, inverted: bool, interp: eInterpolationType, border_mode: eBorderType, border_value: list, stream: rocpycv.Stream | None = None, device: eDeviceType = ...) -> None:
    """
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

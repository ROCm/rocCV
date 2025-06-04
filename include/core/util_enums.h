/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

/* Supported Data Types for use with utilities such as Tensor inputs. */
typedef enum eDataType {
    DATA_TYPE_U8 = 0, /* 8 bit unsigned integer. */
    DATA_TYPE_S8,     /* 8 bit signed integer. */
    DATA_TYPE_U32,    /* 32 bit unsigned integer. */
    DATA_TYPE_S32,    /* 32 bit signed integer. */
    DATA_TYPE_F32,    /* 32 bit single precision float */
    DATA_TYPE_4S16,   /* 4-channel 16-bit signed integer*/
    DATA_TYPE_S16     /* 16 bit signed integer */
} eDataType;

/* Supported Tensor Layout Types. */
typedef enum eTensorLayout {
    TENSOR_LAYOUT_NHWC,  /* Number of Samples, Height, Width, Channels. */
    TENSOR_LAYOUT_HWC,   /* Height, Width, Channels. */
    TENSOR_LAYOUT_NC,    /* Number of Samples, Channels */
    TENSOR_LAYOUT_NW,    /* Number of Samples, Width */
    TENSOR_LAYOUT_N,     /* Number of Samples */
    TENSOR_LAYOUT_NMC,   /* (SIFT) Number of Samples, maximum number of features,
                            number of feature coordinates/metadata */
    TENSOR_LAYOUT_NMD,   /* (SIFT) Number of Samples, maximum number of features,
                            Depth of each feature descriptor*/
    TENSOR_LAYOUT_LNHWC, /* (SIFT) Octave layer, Number of samples, Height,
                            Width, Channels */
    TENSOR_LAYOUT_NCHW,  /* Number of Samples, Channels, Height, Width */
    TENSOR_LAYOUT_NWC,   /* Number of Samples, Width, Channels */
} eTensorLayout;

typedef enum eChannelType {
    F_RGB = 1,
    BGR = 2,
    YUV = 4, /*Treated as NV12 for Semi-Planar images*/
    YVU = 8, /*Treated as NV21 for Semi-Planar images*/
    Grayscale = 16,

} eChannelType;

/**
 * @brief Describes the device type. Used to determine where Tensor data should
 * be allocated and whether operations should run on the host or device (GPU or
 * CPU).
 *
 */
enum class eDeviceType { GPU = 0, CPU = 1 };

/**
 * @brief Describes whether a container should own a resource or only be a view for a resource.
 *
 */
enum class eOwnership { OWNING = 0, VIEW = 1 };

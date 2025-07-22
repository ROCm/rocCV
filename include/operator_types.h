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

#include <hip/hip_vector_types.h>
#include <stdint.h>

#include <vector>

typedef enum eInterpolationType {
    INTERP_TYPE_NEAREST = 0,
    INTERP_TYPE_LINEAR = 1,
    INTERP_TYPE_CUBIC = 2
} eInterpolationType;

typedef enum eBorderType {
    BORDER_TYPE_CONSTANT = 0,   ///< Uses a constant value for borders.
    BORDER_TYPE_REPLICATE = 1,  ///< Replicates the last element for borders.
    BORDER_TYPE_REFLECT = 2,    ///< Reflects the border elements.
    BORDER_TYPE_WRAP = 3,       ///< Wraps the border elements.
} eBorderType;

typedef enum eRemapType {
    REMAP_ABSOLUTE = 0,             
    REMAP_ABSOLUTE_NORMALIZED = 1,
    REMAP_RELATIVE_NORMALIZED = 2,
} eRemapType;

typedef enum eColorConversionCode {
    COLOR_RGB2YUV = 0,
    COLOR_BGR2YUV = 1,
    COLOR_YUV2RGB = 2,
    COLOR_YUV2BGR = 3,
    COLOR_RGB2BGR = 4,
    COLOR_BGR2RGB = 5,
    COLOR_RGB2GRAY = 6,
    COLOR_BGR2GRAY = 7,
} eColorConversionCode;

typedef enum eAxis {
    BOTH = -1,
    X = 0,
    Y = 1,
} eAxis;

typedef enum eColorSpec {
    BT601 = 0,
    BT709 = 1,
    BT2020 = 2,
} eColorSpec;

typedef enum eThresholdType {
    THRESH_BINARY = 0x01,
    THRESH_BINARY_INV = 0x02,
    THRESH_TRUNC = 0x04,
    THRESH_TOZERO = 0x08,
    THRESH_TOZERO_INV = 0x10,
} eThresholdType;

// Column Major
typedef float PerspectiveTransform[9];

typedef struct {
    uint8_t c0;
    uint8_t c1;
    uint8_t c2;
    uint8_t c3;
} Color4_t;

typedef struct {
    int64_t x;       ///@brief x coordinate of the top-left corner.
    int64_t y;       ///@brief y coordinate of the top-left corner.
    int32_t width;   ///@brief width of the box.
    int32_t height;  ///@brief height of the box.
} Box_t;

typedef struct {
    Box_t box;             ///@brief Bounding box definition.
    int32_t thickness;     ///@brief Border thickness of bounding box.
    Color4_t borderColor;  ///@brief Border color of bounding box.
    Color4_t fillColor;    ///@brief Fill color of bounding box.
} BndBox_t;

typedef struct {
    int64_t batch;                  ///@brief Batch size.
    std::vector<int32_t> numBoxes;  ///@brief Vector of number of boxes in each image, must have
                                    /// atleast \ref batch elements.
    std::vector<BndBox_t> boxes;    ///@brief Vector of bounding boxes to draw, must have enough
                                    /// elements to match \ref numBoxes.
} BndBoxes_t;

/**
 * The Rect_t struct is used for the bounding box rectangles for the Bounding Box operator
 */
typedef struct {
    int64_t batch;
    float o_left, o_right, o_bottom, o_top;
    float i_left, i_right, i_bottom, i_top;
    uchar4 color;
    bool bordered;
} Rect_t;


namespace roccv {

/**
 * @brief Describes the 2D dimensions of an image.
 *
 */
struct Size2D {
    int w, h;
};
}  // namespace roccv
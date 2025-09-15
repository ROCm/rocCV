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

/**
 * @brief Describes an 8-bit RGBA color value.
 *
 */
struct ColorRGBA_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

/**
 * @brief Describes a single box.
 *
 */
struct Box_t {
    int32_t x;       // top-left corner x coordinate
    int32_t y;       // top-left corner y coordinate
    int32_t width;   // width of the box
    int32_t height;  // height of the box
};

/**
 * @brief Describes a single bounding box with a border thickness, border color, and fill color.
 *
 */
struct BndBox_t {
    Box_t box;                // bounding box
    int32_t thickness;        // thickness of the box border
    ColorRGBA_t borderColor;  // color of the box border
    ColorRGBA_t fillColor;    // fill color of the bounding box
};

/**
 * @brief Describes a list of bounding boxes to be used alongside the BndBox operator.
 *
 */
class BndBoxes {
   public:
    /**
     * @brief Construct a new BndBoxes object.
     *
     * @param[in] bndboxesVec A list of lists of bounding boxes corresponding to each image in the batch.
     */
    BndBoxes(const std::vector<std::vector<BndBox_t>> &bndboxesVec);
    BndBoxes(const BndBoxes &) = delete;
    BndBoxes &operator=(const BndBoxes &) = delete;

    /**
     * @brief Retrieves the batch size of this bounding box definition.
     *
     * @return The batch size of this bounding box definition.
     */
    int32_t batch() const;

    /**
     * @brief Returns the number of bounding boxes at a specific batch index.
     *
     * @param b
     * @return int32_t
     */
    int32_t numBoxesAt(int32_t b) const;

    /**
     * @brief Returns the bounding box at the specified batch and bounding box index.
     *
     * @param b The batch index.
     * @param i The index of the box within the specified batch.
     * @return A bounding box.
     */
    BndBox_t boxAt(int32_t b, int32_t i) const;

   private:
    std::vector<std::vector<BndBox_t>> m_bndboxesVec;
};

}  // namespace roccv
/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "core/tensor.hpp"
#include "core/exception.hpp"
#include <optional>


typedef enum eInterpolationType {
    INTERP_TYPE_NEAREST = 0,
    INTERP_TYPE_LINEAR = 1,
    INTERP_TYPE_CUBIC = 2
} eInterpolationType;

typedef enum eBorderType {
    BORDER_TYPE_CONSTANT = 0,       ///< Uses a constant value for borders.
    BORDER_TYPE_REPLICATE = 1,      ///< Replicates the last element for borders.
    BORDER_TYPE_REFLECT = 2,        ///< Reflects the border elements, including the boundary pixel
    BORDER_TYPE_REFLECT101 = 3,     ///< Reflects the border elements, excluding the boundary pixel
    BORDER_TYPE_WRAP = 4,           ///< Wraps the border elements.
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

    COLOR_YUV2RGB_NV12 = 8,
    COLOR_YUV2BGR_NV12 = 9,
    COLOR_YUV2RGB_NV21 = 10,
    COLOR_YUV2BGR_NV21 = 11,
    COLOR_RGB2YUV_NV12 = 12,
    COLOR_BGR2YUV_NV12 = 13,
    COLOR_RGB2YUV_NV21 = 14,
    COLOR_BGR2YUV_NV21 = 15,
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

// Used to describe params to the Brightness Contrast operator
typedef enum eBCType {
    BC_TYPE_DEFAULT = 0,      ///< Use default value.
    BC_TYPE_BROADCAST = 1,    ///< Broadcast single value to all samples.
    BC_TYPE_PER = 2           ///< Per-sample values.
} eBCType;

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
    int64_t x;       // top-left corner x coordinate
    int64_t y;       // top-left corner y coordinate
    int64_t width;   // width of the box
    int64_t height;  // height of the box
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
    int64_t batch() const;

    /**
     * @brief Returns the number of bounding boxes at a specific batch index.
     *
     * @param b The batch index.
     * @return The number of boxes at the specified batch index.
     */
    int64_t numBoxesAt(int64_t b) const;

    /**
     * @brief Returns the bounding box at the specified batch and bounding box index.
     *
     * @param b The batch index.
     * @param i The index of the box within the specified batch.
     * @return A bounding box.
     */
    BndBox_t boxAt(int64_t b, int64_t i) const;

   private:
    std::vector<std::vector<BndBox_t>> m_bndboxesVec;
};

/**
 * @brief Wraps parameters to the Brightness Contrast operator for efficient access.
 *
 * @tparam DT Data type of the parameter.
 */
template <typename DT>
class BCWrapper {
   public:
    /**
     * @brief Construct a new BCWrapper object.
     *
     * @param[in] tensor_opt Optional reference to a 1D tensor containing the parameter values.
     * @param[in] default_val Default value to use (one for all samples) when tensor is not provided.
     */
    BCWrapper(std::optional<std::reference_wrapper<const Tensor>> tensor_opt,
              DT default_val) : default_value(default_val) {
        if (!tensor_opt.has_value()) {
            arg_type = eBCType::BC_TYPE_DEFAULT;
            data = nullptr;
            batch_stride = -1;
        } else {
            const Tensor& tensor = tensor_opt->get();
            if (tensor.layout() != eTensorLayout::TENSOR_LAYOUT_N) {
                throw Exception("The given tensor layout is not supported for BCWrapper", eStatusType::NOT_IMPLEMENTED);
            }
            arg_type = (tensor.shape(tensor.layout().batch_index()) == 1) ? eBCType::BC_TYPE_BROADCAST : eBCType::BC_TYPE_PER;
            TensorDataStrided tdata = tensor.exportData<TensorDataStrided>();
            batch_stride = tdata.stride(tensor.layout().batch_index());
            data = static_cast<unsigned char*>(tdata.basePtr());
        }
    }

    /**
     * @brief Retrieves the parameter value for a specific batch index.
     *
     * @param n The batch index.
     * @return The parameter value at the specified batch index.
     */
    __device__ __host__ const DT at(int64_t n) const {
        switch (arg_type) {
            case eBCType::BC_TYPE_BROADCAST:
                return *(reinterpret_cast<DT*>(data));
            case eBCType::BC_TYPE_PER:
                return *(reinterpret_cast<DT*>(data + (batch_stride * n)));
            default:
                return default_value;
        }
    }

   private:
    DT default_value;
    eBCType arg_type;
    int64_t batch_stride;
    unsigned char* data;
};

/**
 * @brief Groups the four brightness/contrast parameter wrappers.
 *
 * @tparam DT Data type of the parameters.
 */
template <typename DT>
struct GroupBCWrappers {
    using ValueType = DT;

    BCWrapper<DT> brightnessWrapper;
    BCWrapper<DT> contrastWrapper;
    BCWrapper<DT> brightnessShiftWrapper;
    BCWrapper<DT> contrastCenterWrapper;
};


}  // namespace roccv
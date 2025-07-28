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
#include "op_bnd_box.hpp"

#include <hip/hip_runtime.h>
#include <functional>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "core/wrappers/image_wrapper.hpp"
#include "common/math_vector.hpp"
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/tensor.hpp"
#include "kernels/device/bnd_box_device.hpp"
#include "kernels/host/bnd_box_host.hpp"

namespace roccv {
BndBox::BndBox() {}

BndBox::~BndBox() {}

template <bool has_alpha, typename T>
void dispatch_bnd_box_dtype(hipStream_t stream, const Tensor& input, const Tensor& output, std::vector<Rect_t> rects, const eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    auto width = inputWrapper.width();
    auto height = inputWrapper.height();
    auto batch_size = inputWrapper.batches();
    switch (device) {
        case eDeviceType::GPU: {
            const auto blockSize = 32;
            const auto xGridSize = (width + blockSize - 1) / blockSize;
            const auto yGridSize = (height + blockSize - 1) / blockSize;
            const auto zGridSize = batch_size;

            Rect_t *rects_ptr = nullptr;
            const auto n_rects = rects.size();

            if (n_rects > 0) {
                HIP_VALIDATE_NO_ERRORS(hipMallocAsync(&rects_ptr, sizeof(Rect_t) * n_rects, stream));
                HIP_VALIDATE_NO_ERRORS(
                    hipMemcpyAsync(rects_ptr, rects.data(), sizeof(Rect_t) * n_rects, hipMemcpyHostToDevice, stream));
            }
            Kernels::Device::bndbox_kernel<has_alpha, T>
                    <<<dim3(xGridSize, yGridSize, zGridSize), dim3(blockSize, blockSize, 1), 0, stream>>>(
                        inputWrapper, outputWrapper, rects_ptr, n_rects, batch_size, height, width);
            if (n_rects > 0) {
                HIP_VALIDATE_NO_ERRORS(hipFreeAsync(rects_ptr, stream));
            }
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::bndbox_kernel<has_alpha, T>(inputWrapper, outputWrapper, rects.data(), rects.size(), batch_size, height, width);
            break;
        }
    }
}

void BndBox::operator()(hipStream_t stream, const Tensor &input, const Tensor &output,
                        const BndBoxes_t bnd_boxes, eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);

    // Ensure the layout and shapes for the input/output tensor match
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    //const auto i_batch_i = input.shape().layout().batch_index();
    //const auto i_batch = (i_batch_i >= 0) ? input.shape()[i_batch_i] : 1;
    const auto i_height = input.shape()[input.shape().layout().height_index()];
    const auto i_width = input.shape()[input.shape().layout().width_index()];
    const auto i_channels = input.shape()[input.shape().layout().channels_index()];

    if (i_channels != 3 && i_channels != 4) {
        throw Exception("Invalid channel size: tensors must have channel size of 3 or 4.",
                        eStatusType::NOT_IMPLEMENTED);
    }

    auto input_data = input.exportData<roccv::TensorDataStrided>();
    auto output_data = output.exportData<roccv::TensorDataStrided>();

    if (input.dtype().size() != input_data.stride(input_data.shape().layout().channels_index()) ||
        output.dtype().size() != output_data.stride(output_data.shape().layout().channels_index())) {
        throw Exception("Invalid channel stride: channel elements must be contiguous.", eStatusType::NOT_IMPLEMENTED);
    }

    //auto batch_size = i_batch;
    auto height = i_height;
    auto width = i_width;

    std::vector<Rect_t> rects;
    generateRects(rects, bnd_boxes, height, width);

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const std::vector<Rect_t>, const eDeviceType)>, 4>>
        funcs =
        {
            {eDataType::DATA_TYPE_U8, {0, 0, dispatch_bnd_box_dtype<false, uchar3>, dispatch_bnd_box_dtype<true, uchar4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, rects, device);



    /*switch (device) {
        case eDeviceType::GPU: {
            const auto blockSize = 32;
            const auto xGridSize = (width + blockSize - 1) / blockSize;
            const auto yGridSize = (height + blockSize - 1) / blockSize;
            const auto zGridSize = batch_size;

            Rect_t *rects_ptr;
            const auto n_rects = rects.size();

            if (n_rects > 0) {
                HIP_VALIDATE_NO_ERRORS(hipMallocAsync(&rects_ptr, sizeof(Rect_t) * n_rects, stream));
                HIP_VALIDATE_NO_ERRORS(
                    hipMemcpyAsync(rects_ptr, rects.data(), sizeof(Rect_t) * n_rects, hipMemcpyHostToDevice, stream));
            }
            const auto channels = input.shape()[input.shape().layout().channels_index()];

            if (channels == 3) {
                Kernels::Device::bndbox_kernel<false, uchar3>
                    <<<dim3(xGridSize, yGridSize, zGridSize), dim3(blockSize, blockSize, 1), 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), rects_ptr, n_rects, batch_size, height,
                        width);
            } else {
                Kernels::Device::bndbox_kernel<true, uchar4>
                    <<<dim3(xGridSize, yGridSize, zGridSize), dim3(blockSize, blockSize, 1), 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), rects_ptr, n_rects, batch_size, height,
                        width);
            }
            if (n_rects > 0) HIP_VALIDATE_NO_ERRORS(hipFreeAsync(rects_ptr, stream));
            break;
        }
        case eDeviceType::CPU: {
            const auto channels = input.shape()[input.shape().layout().channels_index()];

            if (channels == 3) {
                Kernels::Host::bndbox_kernel<false, uchar3>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                            detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                            rects.data(), rects.size(), batch_size, height, width);
            } else {
                Kernels::Host::bndbox_kernel<true, uchar4>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                           detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                           rects.data(), rects.size(), batch_size, height, width);
            }
            break;
        }
    }*/
}

void BndBox::generateRects(std::vector<Rect_t> &rects, const BndBoxes_t &bnd_boxes, int64_t height, int64_t width) {
    if (bnd_boxes.batch == 0) {
        return;
    }

    if (bnd_boxes.numBoxes.empty() || bnd_boxes.boxes.empty()) {
        throw Exception("Invalid BndBoxes_t: has a nullptr but batch != 0.", eStatusType::INVALID_POINTER);
    }

    int32_t total_boxes = 0;
    for (int64_t batch = 0; batch < bnd_boxes.batch; batch++) {
        const auto numBoxes = bnd_boxes.numBoxes[batch];

        for (int32_t i = 0; i < numBoxes; i++) {
            const auto curr_box = bnd_boxes.boxes[total_boxes + i];
            const auto left = std::max(std::min(curr_box.box.x, width - 1), static_cast<int64_t>(0));
            const auto top = std::max(std::min(curr_box.box.y, height - 1), static_cast<int64_t>(0));
            const auto right = std::max(std::min(left + curr_box.box.width - 1, width - 1), static_cast<int64_t>(0));
            const auto bottom = std::max(std::min(top + curr_box.box.height - 1, height - 1), static_cast<int64_t>(0));

            if (left == right || top == bottom || curr_box.box.width <= 0 || curr_box.box.height <= 0) {
                continue;
            }

            if (curr_box.borderColor.c3 == 0 && curr_box.fillColor.c3 == 0) {
                continue;
            }

            // no border
            if (curr_box.thickness == -1 && curr_box.borderColor.c3 != 0) {
                Rect_t rect;

                rect.batch = batch;
                rect.bordered = false;
                rect.color.x = curr_box.borderColor.c0;
                rect.color.y = curr_box.borderColor.c1;
                rect.color.z = curr_box.borderColor.c2;
                rect.color.w = curr_box.borderColor.c3;

                rect.o_left = left;
                rect.o_right = right;
                rect.o_top = top;
                rect.o_bottom = bottom;

                rects.push_back(rect);
            } else if (curr_box.thickness >= 0) {
                // fill rect
                {
                    Rect_t rect;

                    rect.batch = batch;
                    rect.bordered = false;
                    rect.color.x = curr_box.fillColor.c0;
                    rect.color.y = curr_box.fillColor.c1;
                    rect.color.z = curr_box.fillColor.c2;
                    rect.color.w = curr_box.fillColor.c3;

                    rect.o_left = left;
                    rect.o_right = right;
                    rect.o_top = top;
                    rect.o_bottom = bottom;

                    rects.push_back(rect);
                }
                // border rect
                {
                    Rect_t rect;

                    rect.batch = batch;
                    rect.bordered = true;

                    float half_thickness = curr_box.thickness / 2.0f;

                    rect.o_left = left - half_thickness;
                    rect.o_right = right + half_thickness;
                    rect.o_top = top - half_thickness;
                    rect.o_bottom = bottom + half_thickness;

                    rect.i_left = left + half_thickness;
                    rect.i_right = right - half_thickness;
                    rect.i_top = top + half_thickness;
                    rect.i_bottom = bottom - half_thickness;

                    rect.color.x = curr_box.borderColor.c0;
                    rect.color.y = curr_box.borderColor.c1;
                    rect.color.z = curr_box.borderColor.c2;
                    rect.color.w = curr_box.borderColor.c3;

                    rects.push_back(rect);
                }
            } else {
                continue;
            }
        }

        total_boxes += numBoxes;
    }
}
}  // namespace roccv
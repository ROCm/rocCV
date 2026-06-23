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
#include "op_brightness_contrast.hpp"

#include <hip/hip_runtime.h>

#include <functional>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/brightness_contrast_device.hpp"
#include "kernels/host/brightness_contrast_host.hpp"

namespace {
using namespace roccv;
typedef enum eBCType {
    BC_TYPE_DEFAULT = 0,    ///< Use default value.
    BC_TYPE_BROADCAST = 1,  ///< Broadcast single value to all samples.
    BC_TYPE_PER = 2         ///< Per-sample values.
} eBCType;

template <typename DT>
class BCWrapper {
   public:
    BCWrapper(std::optional<std::reference_wrapper<const Tensor>> tensor_opt, DT default_val)
        : default_value(default_val) {
        if (!tensor_opt.has_value()) {
            arg_type = eBCType::BC_TYPE_DEFAULT;
            data = nullptr;
            batch_stride = -1;
        } else {
            const Tensor &tensor = tensor_opt->get();
            if (tensor.layout() != eTensorLayout::TENSOR_LAYOUT_N) {
                throw Exception("The given tensor layout is not supported for BCWrapper", eStatusType::NOT_IMPLEMENTED);
            }
            arg_type =
                (tensor.shape(tensor.layout().batch_index()) == 1) ? eBCType::BC_TYPE_BROADCAST : eBCType::BC_TYPE_PER;
            TensorDataStrided tdata = tensor.exportData<TensorDataStrided>();
            batch_stride = tdata.stride(tensor.layout().batch_index());
            data = static_cast<unsigned char *>(tdata.basePtr());
        }
    }

    __device__ __host__ const DT at(int64_t n) const {
        switch (arg_type) {
            case eBCType::BC_TYPE_BROADCAST:
                return *(reinterpret_cast<DT *>(data));
            case eBCType::BC_TYPE_PER:
                return *(reinterpret_cast<DT *>(data + (batch_stride * n)));
            default:
                return default_value;
        }
    }

   private:
    DT default_value;
    eBCType arg_type;
    int64_t batch_stride;
    unsigned char *data;
};

template <typename DT>
struct GroupBCWrappers {
    using ValueType = DT;

    BCWrapper<DT> brightnessWrapper;
    BCWrapper<DT> contrastWrapper;
    BCWrapper<DT> brightnessShiftWrapper;
    BCWrapper<DT> contrastCenterWrapper;
};
}  // namespace

namespace roccv {

template <typename BCWrappers, typename SRC_DT, typename DST_DT, int NC>
void dispatch_brightness_contrast_channels(hipStream_t stream, const Tensor &input, const Tensor &output,
                                           const BCWrappers &bc_wrappers, eDeviceType device) {
    using SRC_DT_NC = detail::MakeType<SRC_DT, NC>;
    using DST_DT_NC = detail::MakeType<DST_DT, NC>;

    ImageWrapper<SRC_DT_NC> inputWrapper(input);
    ImageWrapper<DST_DT_NC> outputWrapper(output);

    // Launch CPU/GPU kernel depending on requested device type.
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block = Kernels::Device::PackedBlock();
            dim3 grid = Kernels::Device::PackedGrid<DST_DT_NC>(outputWrapper.width(), outputWrapper.height(),
                                                               outputWrapper.batches(), block);
            Kernels::Device::brightness_contrast<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, bc_wrappers);
            break;
        }
        case eDeviceType::CPU: {
            Kernels::Host::brightness_contrast(inputWrapper, outputWrapper, bc_wrappers);
            break;
        }
    }
}

template <typename BCWrappers, typename SRC_DT, typename DST_DT>
void dispatch_brightness_contrast_output_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                               const BCWrappers &bc_wrappers, eDeviceType device) {
    int64_t channels = output.shape(output.layout().channels_index());
    // Select kernel dispatcher based on number of channels.
    // clang-format off
    static const std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const BCWrappers &, eDeviceType)>, 4>
        funcs = {dispatch_brightness_contrast_channels<BCWrappers, SRC_DT, DST_DT, 1>, dispatch_brightness_contrast_channels<BCWrappers, SRC_DT, DST_DT, 2>, dispatch_brightness_contrast_channels<BCWrappers, SRC_DT, DST_DT, 3>, dispatch_brightness_contrast_channels<BCWrappers, SRC_DT, DST_DT, 4>};
    // clang-format on

    auto func = funcs.at(channels - 1);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, bc_wrappers, device);
}

template <typename BCWrappers, typename SRC_DT>
void dispatch_brightness_contrast_input_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                              const BCWrappers &bc_wrappers, eDeviceType device) {
    eDataType output_dtype = output.dtype().etype();

    // Select kernel dispatcher based on a base output datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor &, const Tensor &, const BCWrappers &, eDeviceType)>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  dispatch_brightness_contrast_output_dtype<BCWrappers, SRC_DT, uchar>},
            {eDataType::DATA_TYPE_U16, dispatch_brightness_contrast_output_dtype<BCWrappers, SRC_DT, ushort>},
            {eDataType::DATA_TYPE_S16, dispatch_brightness_contrast_output_dtype<BCWrappers, SRC_DT, short>},
            {eDataType::DATA_TYPE_S32, dispatch_brightness_contrast_output_dtype<BCWrappers, SRC_DT, int>},
            {eDataType::DATA_TYPE_F32, dispatch_brightness_contrast_output_dtype<BCWrappers, SRC_DT, float>},
        };
    // clang-format on
    auto func = funcs.at(output_dtype);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, bc_wrappers, device);
}

template <typename BCWrappers>
void dispatch_brightness_contrast_bc_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                           const BCWrappers &bc_wrappers, eDeviceType device) {
    eDataType input_dtype = input.dtype().etype();

    // Select kernel dispatcher based on a base input datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor &, const Tensor &, const BCWrappers &, eDeviceType)>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  dispatch_brightness_contrast_input_dtype<BCWrappers, uchar>},
            {eDataType::DATA_TYPE_U16, dispatch_brightness_contrast_input_dtype<BCWrappers, ushort>},
            {eDataType::DATA_TYPE_S16, dispatch_brightness_contrast_input_dtype<BCWrappers, short>},
            {eDataType::DATA_TYPE_S32, dispatch_brightness_contrast_input_dtype<BCWrappers, int>},
            {eDataType::DATA_TYPE_F32, dispatch_brightness_contrast_input_dtype<BCWrappers, float>},
        };
    // clang-format on
    auto func = funcs.at(input_dtype);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, bc_wrappers, device);
}

void BrightnessContrast::operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output,
                                    std::optional<std::reference_wrapper<const Tensor>> brightness,
                                    std::optional<std::reference_wrapper<const Tensor>> contrast,
                                    std::optional<std::reference_wrapper<const Tensor>> brightnessShift,
                                    std::optional<std::reference_wrapper<const Tensor>> contrastCenter,
                                    eDeviceType device) const {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 2, 3, 4);

    // Validate output tensor
    CHECK_TENSOR_DATATYPES(output, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Validate brightness/contrast params
    eDataType input_dtype = input.dtype().etype();
    int64_t input_batch = input.shape(input.layout().batch_index());
    eDataType output_dtype = output.dtype().etype();
    eDataType bc_dtype = (input_dtype == eDataType::DATA_TYPE_S32 || output_dtype == eDataType::DATA_TYPE_S32)
                             ? eDataType::DATA_TYPE_F64
                             : eDataType::DATA_TYPE_F32;

    auto validate_bc_param = [&](const auto &param_opt) {
        if (param_opt.has_value()) {
            const Tensor &param = param_opt->get();
            CHECK_TENSOR_COMPARISON(param.dtype().etype() == bc_dtype);
            CHECK_TENSOR_LAYOUT(param, TENSOR_LAYOUT_N);
            CHECK_TENSOR_COMPARISON(param.shape(param.layout().batch_index()) == 1 ||
                                    param.shape(param.layout().batch_index()) == input_batch);
            CHECK_TENSOR_COMPARISON(input.device() == param.device());
        }
    };

    validate_bc_param(brightness);
    validate_bc_param(contrast);
    validate_bc_param(brightnessShift);
    validate_bc_param(contrastCenter);

    auto compute_cc_default = [&]() -> double {
        switch (input_dtype) {
            case eDataType::DATA_TYPE_U8:
                return 1u << (8 - 1);
            case eDataType::DATA_TYPE_U16:
                return 1u << (16 - 1);
            case eDataType::DATA_TYPE_S16:
                return 1u << (16 - 2);
            case eDataType::DATA_TYPE_S32:
                return 1u << (32 - 2);
            case eDataType::DATA_TYPE_F32:
                return 0.5;
            default:
                return 0.5;
        }
    };

    // Select kernel dispatcher based on BC datatype
    if (bc_dtype == eDataType::DATA_TYPE_F32) {
        GroupBCWrappers<float> wrappers{BCWrapper<float>(brightness, 1.0f), BCWrapper<float>(contrast, 1.0f),
                                        BCWrapper<float>(brightnessShift, 0.0f),
                                        BCWrapper<float>(contrastCenter, static_cast<float>(compute_cc_default()))};
        dispatch_brightness_contrast_bc_dtype<GroupBCWrappers<float>>(stream, input, output, wrappers, device);
    } else if (bc_dtype == eDataType::DATA_TYPE_F64) {
        GroupBCWrappers<double> wrappers{BCWrapper<double>(brightness, 1.0), BCWrapper<double>(contrast, 1.0),
                                         BCWrapper<double>(brightnessShift, 0.0),
                                         BCWrapper<double>(contrastCenter, compute_cc_default())};
        dispatch_brightness_contrast_bc_dtype<GroupBCWrappers<double>>(stream, input, output, wrappers, device);
    } else {
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}
}  // namespace roccv
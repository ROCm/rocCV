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
    BC_TYPE_DEFAULT = -1,       ///< Use default value.
    BC_TYPE_BROADCAST_PER = 0,  ///< Use passed in broadcast or per-sample value/s
    BC_TYPE_BROADCAST = 1,      ///< Broadcast single value to all samples.
    BC_TYPE_PER = 2             ///< Per-sample values.
} eBCType;

template <typename DT, eBCType BCType>
class BCWrapper {
   public:
    BCWrapper(std::optional<std::reference_wrapper<const Tensor>> tensor_opt, DT default_value, bool per)
        : m_default_value(default_value), m_data(nullptr) {
        if constexpr (BCType != eBCType::BC_TYPE_DEFAULT) {
            const Tensor &tensor = tensor_opt->get();
            TensorDataStrided tdata = tensor.exportData<TensorDataStrided>();
            if (per) {
                m_batch_stride = tdata.stride(tensor.layout().batch_index());
            }
            m_data = static_cast<const unsigned char *>(tdata.basePtr());
        }
    }

    __device__ __host__ const DT at(int64_t n) const {
        if constexpr (BCType == eBCType::BC_TYPE_BROADCAST_PER) {
            return *(reinterpret_cast<const DT *>(m_data + (m_batch_stride * n)));
        }
        return m_default_value;
    }

   private:
    DT m_default_value;
    int64_t m_batch_stride = 0;
    const unsigned char *m_data;
};

template <typename DT, eBCType B_T, eBCType C_T, eBCType BS_T, eBCType CC_T>
struct GroupBCWrappers {
    using ValueType = DT;

    BCWrapper<DT, B_T> brightnessWrapper;
    BCWrapper<DT, C_T> contrastWrapper;
    BCWrapper<DT, BS_T> brightnessShiftWrapper;
    BCWrapper<DT, CC_T> contrastCenterWrapper;
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
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
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
void dispatch_contrast_center(hipStream_t stream, const Tensor &input, const Tensor &output,
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

template <typename T, eBCType B_T, eBCType C_T, eBCType BS_T>
void dispatch_brightness_shift(hipStream_t stream, const Tensor &input, const Tensor &output,
                               BCWrapper<T, B_T> brightnessWrapper, BCWrapper<T, C_T> contrastWrapper,
                               BCWrapper<T, BS_T> brightnessShiftWrapper,
                               std::optional<std::reference_wrapper<const Tensor>> contrastCenter,
                               eBCType contrastCenterType, eDeviceType device) {
    auto compute_cc_default = [&]() -> double {
        switch (input.dtype().etype()) {
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
    T defaultValue = static_cast<T>(compute_cc_default());
    switch (contrastCenterType) {
        case BC_TYPE_BROADCAST:
            dispatch_contrast_center<GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_BROADCAST_PER>>(
                stream, input, output,
                GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_BROADCAST_PER>{
                    brightnessWrapper,
                    contrastWrapper,
                    brightnessShiftWrapper,
                    BCWrapper<T, BC_TYPE_BROADCAST_PER>(contrastCenter, defaultValue, false),
                },
                device);
            break;
        case BC_TYPE_PER:
            dispatch_contrast_center<GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_BROADCAST_PER>>(
                stream, input, output,
                GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_BROADCAST_PER>{
                    brightnessWrapper,
                    contrastWrapper,
                    brightnessShiftWrapper,
                    BCWrapper<T, BC_TYPE_BROADCAST_PER>(contrastCenter, defaultValue, true),
                },
                device);
            break;
        case BC_TYPE_DEFAULT:
            dispatch_contrast_center<GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_DEFAULT>>(
                stream, input, output,
                GroupBCWrappers<T, B_T, C_T, BS_T, BC_TYPE_DEFAULT>{
                    brightnessWrapper,
                    contrastWrapper,
                    brightnessShiftWrapper,
                    BCWrapper<T, BC_TYPE_DEFAULT>(contrastCenter, defaultValue, false),
                },
                device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}

template <typename T, eBCType B_T, eBCType C_T>
void dispatch_contrast(hipStream_t stream, const Tensor &input, const Tensor &output,
                       BCWrapper<T, B_T> brightnessWrapper, BCWrapper<T, C_T> contrastWrapper,
                       std::optional<std::reference_wrapper<const Tensor>> brightnessShift,
                       std::optional<std::reference_wrapper<const Tensor>> contrastCenter, eBCType brightnessShiftType,
                       eBCType contrastCenterType, eDeviceType device) {
    T defaultValue = static_cast<T>(0.0);
    switch (brightnessShiftType) {
        case BC_TYPE_BROADCAST:
            dispatch_brightness_shift<T, B_T, C_T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, brightnessWrapper, contrastWrapper,
                BCWrapper<T, BC_TYPE_BROADCAST_PER>(brightnessShift, defaultValue, false), contrastCenter,
                contrastCenterType, device);
            break;
        case BC_TYPE_PER:
            dispatch_brightness_shift<T, B_T, C_T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, brightnessWrapper, contrastWrapper,
                BCWrapper<T, BC_TYPE_BROADCAST_PER>(brightnessShift, defaultValue, true), contrastCenter,
                contrastCenterType, device);
            break;
        case BC_TYPE_DEFAULT:
            dispatch_brightness_shift<T, B_T, C_T, BC_TYPE_DEFAULT>(
                stream, input, output, brightnessWrapper, contrastWrapper,
                BCWrapper<T, BC_TYPE_DEFAULT>(brightnessShift, defaultValue, false), contrastCenter, contrastCenterType,
                device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}

template <typename T, eBCType B_T>
void dispatch_brightness(hipStream_t stream, const Tensor &input, const Tensor &output,
                         BCWrapper<T, B_T> brightnessWrapper,
                         std::optional<std::reference_wrapper<const Tensor>> contrast,
                         std::optional<std::reference_wrapper<const Tensor>> brightnessShift,
                         std::optional<std::reference_wrapper<const Tensor>> contrastCenter, eBCType contrastType,
                         eBCType brightnessShiftType, eBCType contrastCenterType, eDeviceType device) {
    T defaultValue = static_cast<T>(1.0);
    switch (contrastType) {
        case BC_TYPE_BROADCAST:
            dispatch_contrast<T, B_T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, brightnessWrapper,
                BCWrapper<T, BC_TYPE_BROADCAST_PER>(contrast, defaultValue, false), brightnessShift, contrastCenter,
                brightnessShiftType, contrastCenterType, device);
            break;
        case BC_TYPE_PER:
            dispatch_contrast<T, B_T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, brightnessWrapper,
                BCWrapper<T, BC_TYPE_BROADCAST_PER>(contrast, defaultValue, true), brightnessShift, contrastCenter,
                brightnessShiftType, contrastCenterType, device);
            break;
        case BC_TYPE_DEFAULT:
            dispatch_contrast<T, B_T, BC_TYPE_DEFAULT>(
                stream, input, output, brightnessWrapper, BCWrapper<T, BC_TYPE_DEFAULT>(contrast, defaultValue, false),
                brightnessShift, contrastCenter, brightnessShiftType, contrastCenterType, device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}

template <typename T>
void dispatch_bc_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                       std::optional<std::reference_wrapper<const Tensor>> brightness,
                       std::optional<std::reference_wrapper<const Tensor>> contrast,
                       std::optional<std::reference_wrapper<const Tensor>> brightnessShift,
                       std::optional<std::reference_wrapper<const Tensor>> contrastCenter, eBCType brightnessType,
                       eBCType contrastType, eBCType brightnessShiftType, eBCType contrastCenterType,
                       eDeviceType device) {
    T defaultValue = static_cast<T>(1.0);
    switch (brightnessType) {
        case BC_TYPE_BROADCAST:
            dispatch_brightness<T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, BCWrapper<T, BC_TYPE_BROADCAST_PER>(brightness, defaultValue, false), contrast,
                brightnessShift, contrastCenter, contrastType, brightnessShiftType, contrastCenterType, device);
            break;
        case BC_TYPE_PER:
            dispatch_brightness<T, BC_TYPE_BROADCAST_PER>(
                stream, input, output, BCWrapper<T, BC_TYPE_BROADCAST_PER>(brightness, defaultValue, true), contrast,
                brightnessShift, contrastCenter, contrastType, brightnessShiftType, contrastCenterType, device);
            break;
        case BC_TYPE_DEFAULT:
            dispatch_brightness<T, BC_TYPE_DEFAULT>(
                stream, input, output, BCWrapper<T, BC_TYPE_DEFAULT>(brightness, defaultValue, false), contrast,
                brightnessShift, contrastCenter, contrastType, brightnessShiftType, contrastCenterType, device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
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

    // Determine brightness/contrast etc. types
    auto determine_bc_type = [](const std::optional<std::reference_wrapper<const Tensor>> &tensor_opt) -> eBCType {
        if (!tensor_opt.has_value()) {
            return eBCType::BC_TYPE_DEFAULT;
        } else {
            const Tensor &tensor = tensor_opt->get();
            return (tensor.shape(tensor.layout().batch_index()) == 1) ? eBCType::BC_TYPE_BROADCAST
                                                                      : eBCType::BC_TYPE_PER;
        }
    };

    eBCType brightnessType = determine_bc_type(brightness);
    eBCType contrastType = determine_bc_type(contrast);
    eBCType brightnessShiftType = determine_bc_type(brightnessShift);
    eBCType contrastCenterType = determine_bc_type(contrastCenter);

    // dispatch based on datatype of BC params
    switch (bc_dtype) {
        case eDataType::DATA_TYPE_F32:
            dispatch_bc_dtype<float>(stream, input, output, brightness, contrast, brightnessShift, contrastCenter,
                                     brightnessType, contrastType, brightnessShiftType, contrastCenterType, device);
            break;
        case eDataType::DATA_TYPE_F64:
            dispatch_bc_dtype<double>(stream, input, output, brightness, contrast, brightnessShift, contrastCenter,
                                      brightnessType, contrastType, brightnessShiftType, contrastCenterType, device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}
}  // namespace roccv
# ##############################################################################
# Copyright (c)  - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

import pytest
import rocpycv

from test_helpers import generate_tensor, compare_tensors


INTERLEAVED_444_CODES = [
    rocpycv.eColorConversionCode.COLOR_RGB2YUV,
    rocpycv.eColorConversionCode.COLOR_BGR2YUV,
    rocpycv.eColorConversionCode.COLOR_YUV2RGB,
    rocpycv.eColorConversionCode.COLOR_YUV2BGR,
]

INTERLEAVED_TO_SEMIPLANAR_CODES = [
    rocpycv.eColorConversionCode.COLOR_RGB2YUV_NV12,
    rocpycv.eColorConversionCode.COLOR_BGR2YUV_NV12,
    rocpycv.eColorConversionCode.COLOR_RGB2YUV_NV21,
    rocpycv.eColorConversionCode.COLOR_BGR2YUV_NV21,
]

SEMIPLANAR_TO_INTERLEAVED_CODES = [
    rocpycv.eColorConversionCode.COLOR_YUV2RGB_NV12,
    rocpycv.eColorConversionCode.COLOR_YUV2BGR_NV12,
    rocpycv.eColorConversionCode.COLOR_YUV2RGB_NV21,
    rocpycv.eColorConversionCode.COLOR_YUV2BGR_NV21,
]

SPECS = [
    rocpycv.eColorSpec.BT601,
    rocpycv.eColorSpec.BT709,
    rocpycv.eColorSpec.BT2020,
]


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8])
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("code", INTERLEAVED_444_CODES)
@pytest.mark.parametrize("samples,width,height", [[1, 64, 48], [2, 128, 72]])
def test_op_advcvtcolor_interleaved444(samples, height, width, code, spec, dtype, device):
    input_tensor = generate_tensor(samples, width, height, 3, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, 3], dtype, rocpycv.eTensorLayout.NHWC, device)

    stream = rocpycv.Stream()
    output = rocpycv.advcvtcolor(input_tensor, code, spec, stream, device)
    rocpycv.advcvtcolor_into(output_golden, input_tensor, code, spec, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8])
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("code", INTERLEAVED_TO_SEMIPLANAR_CODES)
@pytest.mark.parametrize("samples,width,height", [[1, 64, 48], [2, 128, 72]])
def test_op_advcvtcolor_interleaved_to_semiplanar(samples, height, width, code, spec, dtype, device):
    input_tensor = generate_tensor(samples, width, height, 3, dtype, device)
    output_golden = rocpycv.Tensor([samples, (height * 3) // 2, width, 1], dtype, rocpycv.eTensorLayout.NHWC, device)

    stream = rocpycv.Stream()
    output = rocpycv.advcvtcolor(input_tensor, code, spec, stream, device)
    rocpycv.advcvtcolor_into(output_golden, input_tensor, code, spec, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)


@pytest.mark.parametrize("device", [rocpycv.eDeviceType.GPU, rocpycv.eDeviceType.CPU])
@pytest.mark.parametrize("dtype", [rocpycv.eDataType.U8])
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("code", SEMIPLANAR_TO_INTERLEAVED_CODES)
@pytest.mark.parametrize("samples,width,height", [[1, 64, 48], [2, 128, 72]])
def test_op_advcvtcolor_semiplanar_to_interleaved(samples, height, width, code, spec, dtype, device):
    input_tensor = generate_tensor(samples, width, (height * 3) // 2, 1, dtype, device)
    output_golden = rocpycv.Tensor([samples, height, width, 3], dtype, rocpycv.eTensorLayout.NHWC, device)

    stream = rocpycv.Stream()
    output = rocpycv.advcvtcolor(input_tensor, code, spec, stream, device)
    rocpycv.advcvtcolor_into(output_golden, input_tensor, code, spec, stream, device)
    stream.synchronize()

    compare_tensors(output, output_golden)

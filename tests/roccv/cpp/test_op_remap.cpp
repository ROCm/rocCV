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

#include <core/hip_assert.h>

#include <filesystem>
#include <iostream>
#include <op_remap.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
namespace fs = std::filesystem;

namespace {
void TestCorrectness(const std::string& input_image, const std::string& expected_image, eDeviceType device) {
    eBorderType border_type = BORDER_TYPE_CONSTANT;
    const float4 border_value = {0, 0, 0, 0};
    eRemapType remap_type = REMAP_ABSOLUTE;
    eInterpolationType interp_type = INTERP_TYPE_NEAREST;

    Tensor input = createTensorFromImage(input_image, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    auto height = input.shape()[input.shape().layout().height_index()];
    auto width = input.shape()[input.shape().layout().width_index()];

    TensorShape map_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), {height, width, 2});
    DataType map_dtype(eDataType::DATA_TYPE_F32);
    Tensor mapTensor(map_shape, map_dtype, device);

    std::vector<float> colRemapTable;
    std::vector<float> rowRemapTable;

    float halfWidth = width / 2;
    for (int i = 0; i < height; i++) {
        int j = 0;
        for (; j < halfWidth; j++) {
            rowRemapTable.push_back(i);
            colRemapTable.push_back(halfWidth - j);
        }
        for (; j < width; j++) {
            rowRemapTable.push_back(i);
            colRemapTable.push_back(j);
        }
    }

    std::vector<float2> mapData(rowRemapTable.size());
    for (int i = 0; i < rowRemapTable.size(); i++) {
        mapData[i] = make_float2(colRemapTable[i], rowRemapTable[i]);
    }

    auto mapTensor_data = mapTensor.exportData<TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        HIP_VALIDATE_NO_ERRORS(hipMemcpy(mapTensor_data.basePtr(), mapData.data(),
                                         rowRemapTable.size() * sizeof(float2), hipMemcpyHostToDevice));

        Remap op;
        op(stream, input, output, mapTensor, interp_type, interp_type, remap_type, false, border_type, border_value,
           device);

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
    } else if (device == eDeviceType::CPU) {
        memcpy(mapTensor_data.basePtr(), mapData.data(), rowRemapTable.size() * sizeof(float2));

        Remap op;
        op(nullptr, input, output, mapTensor, interp_type, interp_type, remap_type, false, border_type, border_value,
           device);
    }

    EXPECT_TEST_STATUS(compareImage(output, expected_image, 1.0f), eTestStatusType::TEST_SUCCESS);
}
}  // namespace

eTestStatusType test_op_remap(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    fs::path test_data_path = fs::path(argv[1]) / "tests" / "ops";
    fs::path test_image_filepath = test_data_path / "test_input.bmp";
    fs::path expected_image_filepath = test_data_path / "expected_remap.bmp";

    try {
        TestCorrectness(test_image_filepath, expected_image_filepath, eDeviceType::GPU);
        TestCorrectness(test_image_filepath, expected_image_filepath, eDeviceType::CPU);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
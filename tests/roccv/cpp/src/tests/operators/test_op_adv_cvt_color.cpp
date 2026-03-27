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

#include <core/detail/casting.hpp>
#include <core/hip_assert.h>
#include <op_adv_cvt_color.hpp>

#include <array>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

struct Coeff {
    float r2y;
    float g2y;
    float b2y;
    float b2u;
    float r2v;

    float v2r;
    float u2g;
    float v2g;
    float u2b;
};

Coeff GetCoeff(eColorSpec spec) {
    switch (spec) {
        case BT601:
            return {0.299f, 0.587f, 0.114f, 0.564334086f, 0.713266762f, 1.402f, -0.344f, -0.714f, 1.772f};
        case BT709:
            return {0.2126f, 0.7152f, 0.0722f, 0.538909248f, 0.63500127f, 1.5748f, -0.187324f, -0.468124f, 1.8556f};
        case BT2020:
            return {0.2627f, 0.6780f, 0.0593f, 0.531519082f, 0.678150007f, 1.4746f, -0.16455f, -0.57135f, 1.8814f};
        default: throw std::runtime_error("Unsupported color spec");
    }
}

bool IsSemiPlanarToInterleaved(eColorConversionCode code) {
    switch (code) {
        case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_NV12:
        case COLOR_YUV2RGB_NV21:
        case COLOR_YUV2BGR_NV21: return true;
        default: return false;
    }
}

bool IsInterleavedToSemiPlanar(eColorConversionCode code) {
    switch (code) {
        case COLOR_RGB2YUV_NV12:
        case COLOR_BGR2YUV_NV12:
        case COLOR_RGB2YUV_NV21:
        case COLOR_BGR2YUV_NV21: return true;
        default: return false;
    }
}

bool IsInterleaved444(eColorConversionCode code) {
    switch (code) {
        case COLOR_RGB2YUV:
        case COLOR_BGR2YUV:
        case COLOR_YUV2RGB:
        case COLOR_YUV2BGR: return true;
        default: return false;
    }
}

bool IsBGRCode(eColorConversionCode code) {
    switch (code) {
        case COLOR_BGR2YUV:
        case COLOR_YUV2BGR:
        case COLOR_YUV2BGR_NV12:
        case COLOR_YUV2BGR_NV21:
        case COLOR_BGR2YUV_NV12:
        case COLOR_BGR2YUV_NV21: return true;
        default: return false;
    }
}

bool IsNV12Code(eColorConversionCode code) {
    switch (code) {
        case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_NV12:
        case COLOR_RGB2YUV_NV12:
        case COLOR_BGR2YUV_NV12: return true;
        default: return false;
    }
}

inline int idx3(int x, int y, int width, int c) {
    return (y * width + x) * 3 + c;
}

std::vector<uint8_t> GoldenInterleaved444(const std::vector<uint8_t> &input, int samples, int width, int height,
                                          eColorConversionCode code, eColorSpec spec) {
    std::vector<uint8_t> output(input.size(), 0);
    const Coeff coeff = GetCoeff(spec);

    for (int n = 0; n < samples; n++) {
        int batchOffset = n * width * height * 3;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = batchOffset + idx3(x, y, width, 0);

                switch (code) {
                    case COLOR_RGB2YUV:
                    case COLOR_BGR2YUV: {
                        float r = input[i + (IsBGRCode(code) ? 2 : 0)];
                        float g = input[i + 1];
                        float b = input[i + (IsBGRCode(code) ? 0 : 2)];

                        float yv = r * coeff.r2y + g * coeff.g2y + b * coeff.b2y;
                        float u = (b - yv) * coeff.b2u + 128.0f;
                        float v = (r - yv) * coeff.r2v + 128.0f;

                        output[i + 0] = roccv::detail::SaturateCast<uint8_t>(yv);
                        output[i + 1] = roccv::detail::SaturateCast<uint8_t>(u);
                        output[i + 2] = roccv::detail::SaturateCast<uint8_t>(v);
                        break;
                    }
                    case COLOR_YUV2RGB:
                    case COLOR_YUV2BGR: {
                        float yv = input[i + 0];
                        float u = input[i + 1];
                        float v = input[i + 2];

                        float r = yv + (v - 128.0f) * coeff.v2r;
                        float g = yv + (u - 128.0f) * coeff.u2g + (v - 128.0f) * coeff.v2g;
                        float b = yv + (u - 128.0f) * coeff.u2b;

                        output[i + (IsBGRCode(code) ? 0 : 2)] = roccv::detail::SaturateCast<uint8_t>(b);
                        output[i + 1] = roccv::detail::SaturateCast<uint8_t>(g);
                        output[i + (IsBGRCode(code) ? 2 : 0)] = roccv::detail::SaturateCast<uint8_t>(r);
                        break;
                    }
                    default: throw std::runtime_error("Invalid interleaved code");
                }
            }
        }
    }

    return output;
}

std::vector<uint8_t> GoldenInterleavedToSemiPlanar(const std::vector<uint8_t> &input, int samples, int width,
                                                   int height, eColorConversionCode code, eColorSpec spec) {
    std::vector<uint8_t> output(samples * width * height * 3 / 2, 0);
    const Coeff coeff = GetCoeff(spec);
    const bool bgr = IsBGRCode(code);
    const int uidx = IsNV12Code(code) ? 0 : 1;

    for (int n = 0; n < samples; n++) {
        int rgbBase = n * width * height * 3;
        int nvBase = n * width * height * 3 / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src = rgbBase + idx3(x, y, width, 0);
                float r = input[src + (bgr ? 2 : 0)];
                float g = input[src + 1];
                float b = input[src + (bgr ? 0 : 2)];

                float yv = r * coeff.r2y + g * coeff.g2y + b * coeff.b2y;
                output[nvBase + y * width + x] = roccv::detail::SaturateCast<uint8_t>(yv);
            }
        }

        for (int y = 0; y < height; y += 2) {
            for (int x = 0; x < width; x += 2) {
                float uSum = 0.0f;
                float vSum = 0.0f;

                for (int yy = 0; yy < 2; yy++) {
                    for (int xx = 0; xx < 2; xx++) {
                        int src = rgbBase + idx3(x + xx, y + yy, width, 0);
                        float r = input[src + (bgr ? 2 : 0)];
                        float g = input[src + 1];
                        float b = input[src + (bgr ? 0 : 2)];

                        float yv = r * coeff.r2y + g * coeff.g2y + b * coeff.b2y;
                        uSum += (b - yv) * coeff.b2u;
                        vSum += (r - yv) * coeff.r2v;
                    }
                }

                int uvRow = height + y / 2;
                int uvCol = x;
                output[nvBase + uvRow * width + uvCol + uidx] =
                    roccv::detail::SaturateCast<uint8_t>(uSum * 0.25f + 128.0f);
                output[nvBase + uvRow * width + uvCol + (1 - uidx)] =
                    roccv::detail::SaturateCast<uint8_t>(vSum * 0.25f + 128.0f);
            }
        }
    }

    return output;
}

std::vector<uint8_t> GoldenSemiPlanarToInterleaved(const std::vector<uint8_t> &input, int samples, int width,
                                                   int height, eColorConversionCode code, eColorSpec spec) {
    std::vector<uint8_t> output(samples * width * height * 3, 0);
    const Coeff coeff = GetCoeff(spec);
    const bool bgr = IsBGRCode(code);
    const int uidx = IsNV12Code(code) ? 0 : 1;

    for (int n = 0; n < samples; n++) {
        int rgbBase = n * width * height * 3;
        int nvBase = n * width * height * 3 / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float yv = input[nvBase + y * width + x];
                int uvRow = height + y / 2;
                int uvCol = x & ~1;
                float u = input[nvBase + uvRow * width + uvCol + uidx];
                float v = input[nvBase + uvRow * width + uvCol + (1 - uidx)];

                float r = yv + (v - 128.0f) * coeff.v2r;
                float g = yv + (u - 128.0f) * coeff.u2g + (v - 128.0f) * coeff.v2g;
                float b = yv + (u - 128.0f) * coeff.u2b;

                int dst = rgbBase + idx3(x, y, width, 0);
                output[dst + (bgr ? 0 : 2)] = roccv::detail::SaturateCast<uint8_t>(b);
                output[dst + 1] = roccv::detail::SaturateCast<uint8_t>(g);
                output[dst + (bgr ? 2 : 0)] = roccv::detail::SaturateCast<uint8_t>(r);
            }
        }
    }

    return output;
}

void TestCorrectness(int samples, int width, int height, eColorConversionCode code, eColorSpec spec,
                     eDeviceType device) {
    Tensor inputTensor = IsSemiPlanarToInterleaved(code)
                             ? Tensor(samples, {width, height * 3 / 2}, FMT_U8, device)
                             : Tensor(samples, {width, height}, FMT_RGB8, device);

    Tensor outputTensor = IsInterleavedToSemiPlanar(code)
                              ? Tensor(samples, {width, height * 3 / 2}, FMT_U8, device)
                              : Tensor(samples, {width, height}, FMT_RGB8, device);

    std::vector<uint8_t> inputData(inputTensor.shape().size());
    FillVector(inputData);
    CopyVectorIntoTensor(inputTensor, inputData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    AdvCvtColor op;
    op(stream, inputTensor, outputTensor, code, spec, device);

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    std::vector<uint8_t> outputActual(outputTensor.shape().size());
    CopyTensorIntoVector(outputActual, outputTensor);

    std::vector<uint8_t> outputGolden;
    if (IsInterleaved444(code)) {
        outputGolden = GoldenInterleaved444(inputData, samples, width, height, code, spec);
    } else if (IsInterleavedToSemiPlanar(code)) {
        outputGolden = GoldenInterleavedToSemiPlanar(inputData, samples, width, height, code, spec);
    } else {
        outputGolden = GoldenSemiPlanarToInterleaved(inputData, samples, width, height, code, spec);
    }

    CompareVectorsNear(outputActual, outputGolden, 0.01);
}

}  // namespace

int main(int argc, char **argv) {
    TEST_CASES_BEGIN();

    std::array<eColorSpec, 3> specs = {BT601, BT709, BT2020};

    for (auto spec : specs) {
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR, spec, eDeviceType::GPU));

        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV_NV12, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV_NV12, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV_NV21, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV_NV21, spec, eDeviceType::GPU));

        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB_NV12, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR_NV12, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB_NV21, spec, eDeviceType::GPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR_NV21, spec, eDeviceType::GPU));

        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR, spec, eDeviceType::CPU));

        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV_NV12, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV_NV12, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_RGB2YUV_NV21, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_BGR2YUV_NV21, spec, eDeviceType::CPU));

        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB_NV12, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR_NV12, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2RGB_NV21, spec, eDeviceType::CPU));
        TEST_CASE(TestCorrectness(2, 64, 48, COLOR_YUV2BGR_NV21, spec, eDeviceType::CPU));
    }

    TEST_CASES_END();
}

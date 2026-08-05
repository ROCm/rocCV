/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Golden-model equivalence test for the device float -> uint8 SaturateCast fast path.
 *
 * On device, SaturateCast<uchar>(float) lowers to the hardware v_cvt_pk_u8_f32 instruction
 * (see CvtPackedSaturateU8 in core/detail/casting.hpp). This test launches that device path over a
 * representative set of float inputs and asserts it produces byte-for-byte identical results to the
 * host (generic clamp-then-IEEE-round) reference path -- which is the golden model the operator
 * GPU-vs-CPU tests rely on.
 */

#include <core/hip_assert.h>
#include <hip/hip_runtime.h>

#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <cstdint>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv::detail;
using namespace roccv::tests;
using namespace roccv;

namespace {

__global__ void SaturateCastScalarKernel(const float* in, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = SaturateCast<unsigned char>(in[i]);
}

__global__ void SaturateCastVec4Kernel(const float4* in, uchar4* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = SaturateCast<uchar4>(in[i]);
}

/**
 * @brief Builds a set of float inputs that exercises the rounding ties, clamp endpoints, negatives, and
 * out-of-range values that the fast path must handle identically to the host reference.
 */
std::vector<float> BuildInputs() {
    std::vector<float> inputs;

    // Dense sweep across and beyond the [0, 255] range at 1/8 steps. This hits every integer, every
    // .5 tie (round-to-even), and the quarter/eighth fractions in between, plus out-of-range tails.
    for (float v = -8.0f; v <= 264.0f; v += 0.125f) inputs.push_back(v);

    // Explicit critical values.
    const float extras[] = {-0.0f,
                            0.0f,
                            0.5f,
                            1.5f,
                            2.5f,
                            126.5f,
                            127.5f,
                            128.5f,
                            253.5f,
                            254.5f,
                            255.0f,
                            255.5f,
                            255.4999f,
                            255.5001f,
                            256.0f,
                            -1.0f,
                            -0.5f,
                            -100.0f,
                            1000.0f,
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::lowest()};
    for (float v : extras) inputs.push_back(v);

    return inputs;
}

/** @brief Runs the scalar float -> uint8 device path and compares it byte-for-byte to the host reference. */
void TestScalar(const std::vector<float>& inputs) {
    const int n = static_cast<int>(inputs.size());

    // Host golden reference (generic clamp-then-IEEE-round path).
    std::vector<unsigned char> reference(n);
    for (int i = 0; i < n; i++) reference[i] = SaturateCast<unsigned char>(inputs[i]);

    float* d_in = nullptr;
    unsigned char* d_out = nullptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_in, n * sizeof(float)));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_out, n * sizeof(unsigned char)));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_in, inputs.data(), n * sizeof(float), hipMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    SaturateCastScalarKernel<<<grid, block, 0, 0>>>(d_in, d_out, n);
    HIP_VALIDATE_NO_ERRORS(hipGetLastError());
    HIP_VALIDATE_NO_ERRORS(hipDeviceSynchronize());

    std::vector<unsigned char> result(n);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(result.data(), d_out, n * sizeof(unsigned char), hipMemcpyDeviceToHost));

    HIP_VALIDATE_NO_ERRORS(hipFree(d_in));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_out));

    CompareVectors(result, reference);
}

/** @brief Runs the float4 -> uchar4 device path and compares it byte-for-byte to the host reference. */
void TestVec4(const std::vector<float>& scalarInputs) {
    // Pack the scalar inputs into float4s (truncating the ragged tail).
    const int n = static_cast<int>(scalarInputs.size()) / 4;

    std::vector<float4> inputs(n);
    for (int i = 0; i < n; i++) {
        inputs[i] =
            float4{scalarInputs[i * 4 + 0], scalarInputs[i * 4 + 1], scalarInputs[i * 4 + 2], scalarInputs[i * 4 + 3]};
    }

    // Host golden reference, flattened to bytes for an exact comparison.
    std::vector<unsigned char> reference(n * 4);
    for (int i = 0; i < n; i++) {
        uchar4 ref = SaturateCast<uchar4>(inputs[i]);
        reference[i * 4 + 0] = ref.x;
        reference[i * 4 + 1] = ref.y;
        reference[i * 4 + 2] = ref.z;
        reference[i * 4 + 3] = ref.w;
    }

    float4* d_in = nullptr;
    uchar4* d_out = nullptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_in, n * sizeof(float4)));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_out, n * sizeof(uchar4)));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_in, inputs.data(), n * sizeof(float4), hipMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    SaturateCastVec4Kernel<<<grid, block, 0, 0>>>(d_in, d_out, n);
    HIP_VALIDATE_NO_ERRORS(hipGetLastError());
    HIP_VALIDATE_NO_ERRORS(hipDeviceSynchronize());

    std::vector<uchar4> resultVec(n);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(resultVec.data(), d_out, n * sizeof(uchar4), hipMemcpyDeviceToHost));

    HIP_VALIDATE_NO_ERRORS(hipFree(d_in));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_out));

    std::vector<unsigned char> result(n * 4);
    for (int i = 0; i < n; i++) {
        result[i * 4 + 0] = resultVec[i].x;
        result[i * 4 + 1] = resultVec[i].y;
        result[i * 4 + 2] = resultVec[i].z;
        result[i * 4 + 3] = resultVec[i].w;
    }

    CompareVectors(result, reference);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    TEST_CASES_BEGIN();

    const std::vector<float> inputs = BuildInputs();

    // Device float -> uint8 fast path must match the host golden model exactly.
    TEST_CASE(TestScalar(inputs));

    // Device float4 -> uchar4 fast path (the packed-store case) must also match exactly.
    TEST_CASE(TestVec4(inputs));

    TEST_CASES_END();
}

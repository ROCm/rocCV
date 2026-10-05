/*
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

// rocCV Flip on GPU device 0 through the C++ API: does a kernel-launch failure surface?
//   launch_probe correct|honest
// Exit 0 when the mode accepts the outcome, 1 otherwise (see ../verdict.py for the modes).
#include <hip/hip_runtime.h>

#include <core/exception.hpp>
#include <core/tensor.hpp>
#include <op_flip.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace roccv;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool honest = argc > 1 && std::strcmp(argv[1], "honest") == 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("OUTCOME: clean_error - no HIP device 0\n");
        return honest ? 0 : 1;
    }
    printf("HIP device 0: %s (%s)\n", prop.name, prop.gcnArchName);
    const int N = 2, H = 5, W = 7;
    std::vector<uint8_t> src(N * H * W), dst(N * H * W, 0xAA), exp(N * H * W);
    for (int i = 0; i < N * H * W; i++) src[i] = static_cast<uint8_t>((i * 3) % 251);
    for (int b = 0; b < N; b++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) exp[(b * H + y) * W + x] = src[(b * H + y) * W + (W - 1 - x)];
    const char* outcome = "correct";
    try {
        TensorShape shape(TensorLayout(TENSOR_LAYOUT_NHWC), {N, H, W, 1});
        Tensor in(shape, DataType(DATA_TYPE_U8), eDeviceType::GPU);
        Tensor out(shape, DataType(DATA_TYPE_U8), eDeviceType::GPU);
        hipStream_t s = nullptr;
        if (hipStreamCreate(&s) != hipSuccess) {
            printf("OUTCOME: clean_error - hipStreamCreate failed\n");
            return honest ? 0 : 1;
        }
        in.copyFromHostAsync(src.data(), s);
        Flip op;
        op(s, in, out, 1, eDeviceType::GPU);
        hipError_t last = hipGetLastError();
        out.copyToHostAsync(dst.data(), s);
        hipError_t sync = hipStreamSynchronize(s);
        (void)hipStreamDestroy(s);
        int bad = 0;
        for (size_t i = 0; i < dst.size(); i++) bad += dst[i] != exp[i];
        printf("Flip returned normally; hipGetLastError after launch = %s; hipStreamSynchronize = %s; "
               "mismatches = %d/%zu\n", hipGetErrorName(last), hipGetErrorName(sync), bad, dst.size());
        if (bad) outcome = "wrong";
    } catch (const std::exception& e) {
        printf("rocCV threw: %s\n", e.what());
        outcome = "clean_error";
    }
    const bool ok = std::strcmp(outcome, "correct") == 0 || (honest && std::strcmp(outcome, "clean_error") == 0);
    printf("OUTCOME: %s\nVERDICT: %s (mode=%s)\n", outcome, ok ? "PASS" : "FAIL", honest ? "honest" : "correct");
    return ok ? 0 : 1;
}

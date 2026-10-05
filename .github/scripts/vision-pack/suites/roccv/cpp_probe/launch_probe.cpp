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

// Does an operator surface a kernel-launch failure (M21)? On a targeted GPU the launch succeeds, so this is the
// control for the robustness suite's unsupported-GPU run.
// cpp-probe.launch::flip_launch_status: fail when Flip returns normally while hipGetLastError() reports an error.
#include <hip/hip_runtime.h>

#include <core/tensor.hpp>
#include <op_flip.hpp>
#include <string>

#include "check.hpp"

using namespace roccv;

int main() {
    hipDeviceProp_t prop{};
    (void)hipGetDeviceProperties(&prop, 0);
    TensorShape shape(TensorLayout(TENSOR_LAYOUT_NHWC), {1, 8, 8, 1});
    std::string dev = std::string(prop.name) + " (" + prop.gcnArchName + ")";
    try {
        Tensor in(shape, DataType(DATA_TYPE_U8), eDeviceType::GPU);
        Tensor out(shape, DataType(DATA_TYPE_U8), eDeviceType::GPU);
        Flip op;
        op(nullptr, in, out, 1, eDeviceType::GPU);
    } catch (const std::exception& e) {
        vp_check("cpp-probe.launch", "flip_launch_status", "pass", dev + ": Flip reported the error: " + e.what());
        return 0;
    }
    hipError_t last = hipGetLastError();
    hipError_t sync = hipDeviceSynchronize();
    std::string msg = dev + ": Flip returned normally; hipGetLastError=" + hipGetErrorName(last) +
                      " hipDeviceSynchronize=" + hipGetErrorName(sync);
    vp_check("cpp-probe.launch", "flip_launch_status", last == hipSuccess && sync == hipSuccess ? "pass" : "fail",
             msg + (last == hipSuccess ? "" : " (launch failure swallowed)"));
    return 0;
}

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

// TensorWrapData ownership and device-type probes (H19).
//
//   wrap_probe host   cpp-probe.wrap::host_buffer_device       wrapped host memory must report device CPU
//                     cpp-probe.wrap::host_buffer_cpu_op       a CPU operator must accept the wrapped host tensor
//                     cpp-probe.strided::copyToAsync_<dev>     copy a wrapped strided (ROI) host buffer
//                     cpp-probe.strided::copyToHostAsync_packed
//                     The wrapped tensors are deliberately leaked: destroying them makes rocCV free memory it does
//                     not own, which is the H19 double free this mode must not trigger.
//   wrap_probe gpu    cpp-probe.wrap::caller_owned_gpu_memory  after a wrapped tensor is destroyed, the caller's own
//                     hipFree must still succeed. rocCV frees the caller's memory first, so this is a double free:
//                     a deliberate crash probe, run only where crash tests are allowed.
#include <hip/hip_runtime.h>

#include <core/tensor.hpp>
#include <cstdlib>
#include <cstring>
#include <op_flip.hpp>
#include <vector>

#include "check.hpp"

using namespace roccv;

static const char* devname(eDeviceType d) { return d == eDeviceType::GPU ? "GPU" : "CPU"; }

static int host_mode() {
    const int64_t N = 1, H = 4, W = 8, C = 1;
    TensorShape shape(TensorLayout(TENSOR_LAYOUT_NHWC), {N, H, W, C});
    alignas(64) static uint8_t host[64];
    TensorBuffer hb{};
    hb.strided.basePtr = host;
    hb.strided.strides = {H * W * C, W * C, C, 1};
    TensorDataStridedHost htd(shape, DataType(DATA_TYPE_U8), hb);
    auto* ht = new Tensor(TensorWrapData(htd));
    std::string msg = std::string("TensorDataStridedHost.device()=") + devname(htd.device()) +
                      ", TensorWrapData(host data).device()=" + devname(ht->device());
    vp_check("cpp-probe.wrap", "host_buffer_device", ht->device() == eDeviceType::CPU ? "pass" : "fail", msg);
    try {
        Tensor out(shape, DataType(DATA_TYPE_U8), eDeviceType::CPU);
        Flip op;
        op(nullptr, *ht, out, 1, eDeviceType::CPU);
        vp_check("cpp-probe.wrap", "host_buffer_cpu_op", "pass", "CPU Flip on the wrapped host tensor succeeded");
    } catch (const std::exception& e) {
        vp_check("cpp-probe.wrap", "host_buffer_cpu_op", "fail", std::string("CPU Flip on the wrapped host tensor: ") + e.what());
    }

    const int BN = 2, BH = 10, BW = 12, BC = 3, RH = 5, RW = 6, Y0 = 2, X0 = 3;
    static std::vector<uint8_t> base(BN * BH * BW * BC);
    for (size_t i = 0; i < base.size(); i++) base[i] = static_cast<uint8_t>(i & 0xFF);
    TensorBuffer buf{};
    buf.strided.basePtr = base.data() + (Y0 * BW + X0) * BC;
    buf.strided.strides = {BH * BW * BC, BW * BC, BC, 1};
    TensorShape roiShape(TensorLayout(TENSOR_LAYOUT_NHWC), {BN, RH, RW, BC});
    TensorDataStridedHost td(roiShape, DataType(DATA_TYPE_U8), buf);
    auto* roi = new Tensor(TensorWrapData(td));
    auto expected = [&](int n, int y, int x, int c) { return base[((n * BH + Y0 + y) * BW + X0 + x) * BC + c]; };
    auto count_bad = [&](const std::vector<uint8_t>& got) {
        int bad = 0;
        for (int n = 0; n < BN; n++)
            for (int y = 0; y < RH; y++)
                for (int x = 0; x < RW; x++)
                    for (int c = 0; c < BC; c++) bad += got[((n * RH + y) * RW + x) * BC + c] != expected(n, y, x, c);
        return bad;
    };
    const int total = BN * RH * RW * BC;
    for (eDeviceType dev : {eDeviceType::CPU, eDeviceType::GPU}) {
        std::string id = std::string("copyToAsync_") + devname(dev);
        try {
            Tensor dst(roiShape, DataType(DATA_TYPE_U8), dev);
            roi->copyToAsync(dst, nullptr);
            (void)hipStreamSynchronize(nullptr);
            std::vector<uint8_t> hostv(total);
            dst.copyToHostAsync(hostv.data(), nullptr);
            (void)hipStreamSynchronize(nullptr);
            int bad = count_bad(hostv);
            vp_check("cpp-probe.strided", id, bad ? "fail" : "pass",
                     std::to_string(bad) + "/" + std::to_string(total) + " elements wrong");
        } catch (const std::exception& e) {
            vp_check("cpp-probe.strided", id, "fail", std::string("throws: ") + e.what());
        }
    }
    try {
        std::vector<uint8_t> packed(total);
        roi->copyToHostAsync(packed.data(), nullptr);
        (void)hipStreamSynchronize(nullptr);
        int bad = count_bad(packed);
        vp_check("cpp-probe.strided", "copyToHostAsync_packed", bad ? "fail" : "pass",
                 std::to_string(bad) + "/" + std::to_string(total) + " elements wrong");
    } catch (const std::exception& e) {
        vp_check("cpp-probe.strided", "copyToHostAsync_packed", "fail", std::string("throws: ") + e.what());
    }
    std::fflush(stdout);
    std::_Exit(0);
}

static int gpu_mode() {
    const int64_t N = 1, H = 4, W = 8, C = 1;
    TensorShape shape(TensorLayout(TENSOR_LAYOUT_NHWC), {N, H, W, C});
    void* dptr = nullptr;
    if (hipMalloc(&dptr, N * H * W * C) != hipSuccess) {
        vp_check("cpp-probe.wrap", "caller_owned_gpu_memory", "error", "hipMalloc failed");
        return 0;
    }
    {
        TensorBuffer b{};
        b.strided.basePtr = dptr;
        b.strided.strides = {H * W * C, W * C, C, 1};
        TensorDataStridedHip td(shape, DataType(DATA_TYPE_U8), b);
        Tensor t = TensorWrapData(td);
    }
    hipError_t e = hipFree(dptr);
    vp_check("cpp-probe.wrap", "caller_owned_gpu_memory", e == hipSuccess ? "pass" : "fail",
             std::string("caller hipFree after the wrapped tensor was destroyed: ") + hipGetErrorName(e) +
                 (e == hipSuccess ? "" : " (rocCV already freed caller-owned memory: double free)"));
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "gpu") == 0) return gpu_mode();
    return host_mode();
}

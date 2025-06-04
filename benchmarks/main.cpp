/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <core/hip_assert.h>
#include <hip/hip_runtime.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <roccvbench/registry.hpp>
#include <thread>

/**
 * @brief Gets the CPU name as described in /proc/cpuinfo. This is a Linux specific solution.
 *
 * @return The system's CPU name as a string.
 */
std::string GetCPUName() {
    std::string cpuinfo_path = "/proc/cpuinfo";
    std::ifstream cpuinfo(cpuinfo_path);

    if (!cpuinfo.is_open()) {
        fprintf(stderr, "Unable to open file %s\n", cpuinfo_path.c_str());
        return "None";
    }

    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(":");
            cpuinfo.close();
            return line.substr(pos + 2);
        }
    }

    // Unable to find CPU information.
    return "None";
    cpuinfo.close();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <runs_per_benchmark> [output_dir]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Default result directory if one isn't specified
    std::filesystem::path output_dir("bench_results");
    if (argc >= 3) {
        // User has provided a directory
        output_dir = std::filesystem::path(argv[2]);
    }
    std::filesystem::create_directory(output_dir);

    int runs = std::stoi(argv[1]);
    std::filesystem::path result_path(output_dir / "benchmarks.csv");
    std::filesystem::path device_info_path(output_dir / "device_info.csv");

    const std::vector<roccvbench::BenchmarkConfig> configs = {{1, 1080, 1920, runs},   {10, 1080, 1920, runs},
                                                              {25, 1080, 1920, runs},  {50, 1080, 1920, runs},
                                                              {100, 1080, 1920, runs}, {200, 1080, 1920, runs}};

    std::filesystem::create_directory(output_dir);
    std::ofstream result_stream(result_path);
    if (!result_stream.is_open()) {
        fprintf(stderr, "Unable to open %s for writing.\n", result_path.c_str());
        return EXIT_FAILURE;
    }

    std::ofstream device_info_stream(device_info_path);
    if (!device_info_stream.is_open()) {
        fprintf(stderr, "Unable to open %s for writing.\n", device_info_path.c_str());
        return EXIT_FAILURE;
    }

    // Get device information
    int device_id;
    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_VALIDATE_NO_ERRORS(hipGetDeviceProperties(&props, device_id));

    device_info_stream << "GPU Name,CPU Name,Thread Count" << std::endl;
    device_info_stream << props.name << "," << GetCPUName() << "," << std::thread::hardware_concurrency() << std::endl;
    device_info_stream.close();
    printf("Wrote device information to: %s\n", device_info_path.c_str());

    result_stream << "Category,Name,Batches,Height,Width,Execution Time" << std::endl;

    // Iterate through all collected benchmarks.
    try {
        for (auto benchmark : roccvbench::BenchmarkRegistry::instance().getBenchmarks()) {
            printf("Running benchmark %s::%s:\n", benchmark.category.c_str(), benchmark.name.c_str());

            for (auto config : configs) {
                printf("\tConfig: [batches=%li, height=%li, width=%li, runs=%li]\n", config.batches, config.height,
                       config.width, config.runs);
                auto result = benchmark.func(config);

                // Write results to the output file
                result_stream << benchmark.category << "," << benchmark.name << "," << config.batches << ","
                              << config.height << "," << config.width << "," << result.execution_time << std::endl;
            }
            printf("\n");
        }
    } catch (roccv::Exception e) {
        fprintf(stderr, "Exception during benchmarking: %s\n", e.what());
        result_stream.close();
    }

    result_stream.close();
    printf("Wrote benchmark results to: %s\n", result_path.c_str());

    return 0;
}
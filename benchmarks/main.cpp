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
#include <nlohmann/json.hpp>
#include <roccvbench/registry.hpp>
#include <thread>

/**
 * @brief Gets the CPU name as described in /proc/cpuinfo. This is a Linux specific solution.
 *
 * @return The system's CPU name as a string.
 */
std::string getCPUName() {
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

std::vector<roccvbench::BenchmarkConfig> loadConfig(const std::string& filepath) {
    std::ifstream inFile(filepath);

    if (!inFile.is_open()) {
        std::cerr << "Unable to open benchmark configuration file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Loaded config " << filepath << std::endl;

    nlohmann::json data = nlohmann::json::parse(inFile);
    inFile.close();

    std::vector<roccvbench::BenchmarkConfig> result;

    for (auto benchParams : data["params"]) {
        roccvbench::BenchmarkConfig config;
        config.batches = benchParams["batches"];
        config.height = benchParams["height"];
        config.width = benchParams["width"];
        config.runs = benchParams["runs"];

        result.push_back(config);
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_path> [output_dir]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Default result directory if one isn't specified
    std::filesystem::path output_dir("bench_results");
    if (argc >= 3) {
        // User has provided a directory
        output_dir = std::filesystem::path(argv[2]);
    }

    // Load benchmark configuration file
    std::vector<roccvbench::BenchmarkConfig> configs = loadConfig(argv[1]);

    // Get device information
    int device_id;
    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_VALIDATE_NO_ERRORS(hipGetDeviceProperties(&props, device_id));

    nlohmann::json resultsJson;

    // Write device information to output JSON
    resultsJson["device_info"]["gpu"]["name"] = props.name;
    resultsJson["device_info"]["cpu"]["name"] = getCPUName();
    resultsJson["device_info"]["cpu"]["threads"] = std::thread::hardware_concurrency();

    // Iterate through all collected benchmarks.
    try {
        for (auto benchmark : roccvbench::BenchmarkRegistry::instance().getBenchmarks()) {
            printf("Running benchmark %s::%s:\n", benchmark.category.c_str(), benchmark.name.c_str());

            nlohmann::json runResultsJson;
            runResultsJson["name"] = benchmark.name;

            for (auto config : configs) {
                printf("\tConfig: [batches=%li, height=%li, width=%li, runs=%li]\n", config.batches, config.height,
                       config.width, config.runs);
                auto result = benchmark.func(config);

                // Write run results to output JSON
                runResultsJson["width"].push_back(config.width);
                runResultsJson["height"].push_back(config.height);
                runResultsJson["runs"].push_back(config.runs);
                runResultsJson["execution_time"].push_back(result.execution_time);
                runResultsJson["batches"].push_back(config.batches);
            }
            std::cout << std::endl;

            resultsJson["results"][benchmark.category].push_back(runResultsJson);
        }
    } catch (roccv::Exception e) {
        fprintf(stderr, "Exception during benchmarking: %s\n", e.what());
    }

    // Write benchmark results to disk
    std::filesystem::create_directory(output_dir);
    std::filesystem::path resultPath(output_dir / "benchmarks.json");
    std::ofstream resultFile(resultPath);

    if (!resultFile.is_open()) {
        std::cerr << "Unable to open " << resultPath << " for writing results" << std::endl;
        exit(EXIT_FAILURE);
    }

    resultFile << std::setw(4) << resultsJson << std::endl;
    resultFile.close();

    std::cout << "Wrote benchmark results to " << resultPath << std::endl;

    return 0;
}
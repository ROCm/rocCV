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
 * @file main.cpp
 * @brief Entrypoint program for running the rocCV benchmarking suite. Provides options to run a selection of available
 * benchmarks. Benchmark results are then serialized to JSON.
 */

/**
 * @brief Gets the CPU name as described in /proc/cpuinfo. This is a Linux specific solution.
 *
 * @return The system's CPU name as a string.
 */
std::string getCPUName() {
    std::string cpuinfo_path = "/proc/cpuinfo";
    std::ifstream cpuinfo(cpuinfo_path);

    if (!cpuinfo.is_open()) {
        std::cerr << "Unable to open file: " << cpuinfo_path.c_str() << std::endl;
        return "Unknown";
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
    return "Unknown";
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
        config.samples = benchParams["samples"];
        config.height = benchParams["height"];
        config.width = benchParams["width"];
        config.runs = benchParams["runs"];

        result.push_back(config);
    }

    return result;
}

void printHelp(const char* programName) {
    // clang-format off
    std::cout << "rocCV Benchmark Suite\n";
    std::cout << "Description:\n";
    std::cout << "  Runs a series of benchmarks for rocCV operators and outputs performance results.\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " --config <config_filepath> --output output_dir [options]\n";
    std::cout << "  " << programName << " --list\n";
    std::cout << "  " << programName << " --help\n";
    std::cout << "Required Arguments:\n";
    std::cout << "  --config, -c:                   The JSON configuration file to run the benchmarks.\n";
    std::cout << "Optional Arguments:\n";
    std::cout << "  --output, -o:                   The output JSON filepath for benchmark results. Defaults to\n";
    std::cout << "                                  roccv_bench_results.json.\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h:                     Displays this help message.\n";
    std::cout << "  --list, -l:                     Lists the available benchmark categories and exits the program.\n";
    std::cout << "  --select, -s <cat1,cat2,...>:   Selects categories to run benchmarks on. Must be a comma separated list.\n";
    std::cout << "                                  For example: --select Rotate,Flip\n";
    std::cout << "  --exclude, -e <cat1,cat2,...>:  Excludes categories from the benchmark. Must be a comma separated list.\n";
    std::cout << "                                  For example: --exclude Rotate,Flip\n";
    std::cout << "Examples:\n";
    std::cout << "  1. Run all benchmarks using 'config.json', save results to 'results.json'\n";
    std::cout << "      " << programName << " --config config.json --output results.json\n";
    std::cout << "  2. Run benchmarks for categories Rotate and Flip\n";
    std::cout << "      " << programName << " --config config.json --output results.json --select Rotate,Flip\n";
    std::cout << "  3. Run benchmarks on all categories except Rotate and Flip\n";
    std::cout << "      " << programName << " --config config.json --output results.json --exclude Rotate,Flip\n";
    std::cout << "  4. List available benchmark categories\n";
    std::cout << "      " << programName << " --list\n";
    // clang-format on
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName
              << " --config,-c <config_filepath> [--output,-o <output_filepath>] [options]\n";
}

std::vector<std::string> splitStringByComma(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string segment;

    while (std::getline(ss, segment, ',')) {
        result.push_back(segment);
    }

    return result;
}

int main(int argc, char** argv) {
    std::string configFilepath;
    std::string outputFilepath = "roccv_bench_results.json";
    std::vector<std::string> selectedCategories;
    std::vector<std::string> excludedCategories;

    // Collect all available benchmark categories
    std::vector<std::string> availableCategories;
    for (const auto& [category, _] : roccvbench::BenchmarkRegistry::instance().getBenchmarks()) {
        availableCategories.push_back(category);
    }
    std::sort(availableCategories.begin(), availableCategories.end());

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return EXIT_SUCCESS;
        }

        else if (arg == "-l" || arg == "--list") {
            // List all available categories and exit the program
            std::cout << "Available benchmark categories:" << std::endl;
            for (const auto& category : availableCategories) {
                std::cout << "\t" << category << std::endl;
            }
            return EXIT_SUCCESS;
        }

        else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                configFilepath = argv[++i];
            } else {
                std::cerr << "Error: --config requires a value." << std::endl;
                return EXIT_FAILURE;
            }
        }

        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                outputFilepath = argv[++i];
            } else {
                std::cerr << "Error: --output requires a value." << std::endl;
                return EXIT_FAILURE;
            }
        }

        else if (arg == "-s" || arg == "--select") {
            if (i + 1 < argc) {
                if (!excludedCategories.empty()) {
                    std::cerr << "Error: --exclude and --select cannot both be used.\n";
                    return EXIT_FAILURE;
                }
                selectedCategories = splitStringByComma(argv[++i]);
            } else {
                std::cerr << "Error: --select requires a value." << std::endl;
                return EXIT_FAILURE;
            }
        }

        else if (arg == "-e" || arg == "--exclude") {
            if (i + 1 < argc) {
                if (!selectedCategories.empty()) {
                    std::cerr << "Error: --exclude and --select cannot both be used.\n";
                    return EXIT_FAILURE;
                }
                excludedCategories = splitStringByComma(argv[++i]);
            } else {
                std::cerr << "Error: --exclude requires a value." << std::endl;
                return EXIT_FAILURE;
            }
        }

        else {
            std::cerr << "Error: Unrecognized argument: " << arg << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Validate command line arguments
    if (configFilepath.empty()) {
        std::cerr << "Error: A configuration file must be provided.\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Load benchmark configuration file
    std::vector<roccvbench::BenchmarkConfig> configs = loadConfig(configFilepath);

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

    // Determine the final list of categories to run and store it in selectedCategories.

    if (!selectedCategories.empty()) {
        // --select was used. selectedCategories currently holds the user's raw selection.
        // Intersect the selected and available categories
        std::vector<std::string> userRawSelection = selectedCategories;
        std::sort(userRawSelection.begin(), userRawSelection.end());

        selectedCategories.clear();
        std::set_intersection(userRawSelection.begin(), userRawSelection.end(), availableCategories.begin(),
                              availableCategories.end(), std::back_inserter(selectedCategories));
    } else if (!excludedCategories.empty()) {
        // --exclude used. Exclude categories from available categories
        std::vector<std::string> userRawExclusions = excludedCategories;
        std::sort(userRawExclusions.begin(), userRawExclusions.end());

        std::set_difference(availableCategories.begin(), availableCategories.end(), userRawExclusions.begin(),
                            userRawExclusions.end(), std::back_inserter(selectedCategories));
    } else {
        // Neither --select nor --exclude was used.
        selectedCategories = availableCategories;
    }

    // It is possible for the user to exclude all categories or select only categories which do not exist.
    // In that case, warn the user that no benchmarks are to be run and quit the program without writing anything.
    if (selectedCategories.empty()) {
        std::cout << "Warning: Selected/Excluded categories resulted in no benchmarks being selected. Quitting.\n";
        return EXIT_SUCCESS;
    }

    // Print which categories will be run for the benchmark, based on the user's selections.
    std::cout << "Running benchmarks for the following categories:\n";
    for (const auto& selectedCategory : selectedCategories) {
        std::cout << "\t" << selectedCategory << std::endl;
    }

    // Iterate through all collected benchmarks.
    try {
        // Iterate through each selected category
        for (const auto& selectedCategory : selectedCategories) {
            // Iterate through each benchmark in the category
            for (const auto& benchmark :
                 roccvbench::BenchmarkRegistry::instance().getBenchmarks().at(selectedCategory)) {
                std::cout << "Running benchmark " << benchmark.category << "::" << benchmark.name << std::endl;

                nlohmann::json runResultsJson;
                runResultsJson["name"] = benchmark.name;

                // Iterate through each benchmark configuration
                for (auto config : configs) {
                    std::cout << "\tConfig [samples=" << config.samples << ", height=" << config.height
                              << ", width=" << config.width << ", runs=" << config.runs << "]" << std::endl;
                    auto result = benchmark.func(config);

                    // Write run results to output JSON
                    runResultsJson["width"].push_back(config.width);
                    runResultsJson["height"].push_back(config.height);
                    runResultsJson["runs"].push_back(config.runs);
                    runResultsJson["execution_time"].push_back(result.executionTime);
                    runResultsJson["samples"].push_back(config.samples);
                }
                std::cout << std::endl;

                resultsJson["results"][benchmark.category].push_back(runResultsJson);
            }
        }
    } catch (roccv::Exception e) {
        std::cerr << "Exception during benchmarking: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Write benchmark results to disk
    std::filesystem::path resultPath(outputFilepath);
    std::ofstream resultFile(resultPath);

    if (!resultFile.is_open()) {
        std::cerr << "Unable to open " << resultPath << " for writing results" << std::endl;
        return EXIT_FAILURE;
    }

    resultFile << std::setw(4) << resultsJson << std::endl;
    resultFile.close();

    std::cout << "Wrote benchmark results to " << resultPath << std::endl;

    return 0;
}
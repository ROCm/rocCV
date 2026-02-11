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

#include "roccvbench/serializers.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace roccvbench {

void JsonBenchmarkSerializer::serialize(const Results& results, std::filesystem::path& filepath) {
    nlohmann::json json;

    const auto metadata = results.getMetadata();

    json["device_info"]["cpu"]["name"] = std::get<std::string>(metadata.at("cpu"));
    json["device_info"]["cpu"]["threads"] = std::get<size_t>(metadata.at("cpu_threads"));
    json["device_info"]["gpu"]["name"] = std::get<std::string>(metadata.at("gpu"));

    json["results"] = nlohmann::json::object();

    for (const auto& run : results.getRuns()) {
        const auto& values = run.getValues();
        std::string name = std::get<std::string>(values.at("name"));
        std::string category = std::get<std::string>(values.at("category"));

        if (json["results"].find(category) == json["results"].end()) {
            json["results"][category] = nlohmann::json::object();
        }
        if (json["results"][category].find(name) == json["results"][category].end()) {
            json["results"][category][name] = nlohmann::json::array();
        }

        auto runData = nlohmann::json::object();
        for (const auto& [key, value] : values) {
            runData[key] = std::visit([](const auto& val) -> nlohmann::json { return val; }, value);
        }
        json["results"][category][name].emplace_back(runData);
    }

    std::ofstream file(filepath);
    file << json.dump(4);
    file.close();
}

void CsvBenchmarkSerializer::serialize(const Results& results, std::filesystem::path& filepath) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath.string());
    }

    for (const auto& key : results.getRegisteredKeys()) {
        file << key << ",";
    }
    for (const auto& [key, value] : results.getMetadata()) {
        file << key << ",";
    }

    file << std::endl;

    for (const auto& run : results.getRuns()) {
        for (const auto& key : results.getRegisteredKeys()) {
            std::visit([&file](const auto& val) { file << val << ","; }, run.getValues().at(key));
        }
        for (const auto& [key, value] : results.getMetadata()) {
            std::visit([&file](const auto& val) { file << val << ","; }, value);
        }
        file << std::endl;
    }

    file.close();
}
}  // namespace roccvbench
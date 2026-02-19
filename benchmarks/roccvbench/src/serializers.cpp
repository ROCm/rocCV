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

#include "roccvbench/serializers.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>

namespace roccvbench {

namespace {
/**
 * @brief Writes a value to a CSV file, ensuring that strings are quoted.
 *
 * @param file The file to write the value to.
 * @param value The value to write.
 */
void WriteCsvValue(std::ofstream& file, const BenchValue& value) {
    std::visit(
        [&file](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::string>) {
                file << "\"" << val << "\"";
            } else {
                file << val;
            }
        },
        value);
}
}  // namespace

void JsonBenchmarkSerializer::serialize(const Results& results, const std::filesystem::path& filepath) {
    nlohmann::json json;

    const auto metadata = results.getMetadata();
    auto metadata_json = nlohmann::json::object();
    for (const auto& [key, value] : metadata) {
        metadata_json[key] = std::visit([](const auto& val) -> nlohmann::json { return val; }, value);
    }
    json["metadata"] = metadata_json;

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

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath.string());
    }

    file << json.dump(4);
    file.close();
}

void CsvBenchmarkSerializer::serialize(const Results& results, const std::filesystem::path& filepath) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath.string());
    }

    // Write header
    std::string sep;
    for (const auto& key : results.getRegisteredKeys()) {
        file << sep << key;
        sep = ",";
    }
    for (const auto& [key, value] : results.getMetadata()) {
        file << sep << key;
        sep = ",";
    }

    file << std::endl;

    // Write data rows
    for (const auto& run : results.getRuns()) {
        sep = "";
        for (const auto& key : results.getRegisteredKeys()) {
            if (!run.getValues().contains(key)) {
                file << sep;
                sep = ",";
                continue;
            }
            file << sep;
            WriteCsvValue(file, run.getValues().at(key));
            sep = ",";
        }
        for (const auto& [key, value] : results.getMetadata()) {
            file << sep;
            WriteCsvValue(file, value);
            sep = ",";
        }
        file << std::endl;
    }

    file.close();
}
}  // namespace roccvbench
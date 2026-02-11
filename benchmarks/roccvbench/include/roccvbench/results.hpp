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

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace roccvbench {

using BenchValue = std::variant<double, size_t, std::string>;

/**
 * @brief Class for storing and managing benchmark run data.
 *
 */
class RunData {
   public:
    RunData() = default;
    RunData(const std::map<std::string, BenchValue>& runData) : m_runData(runData) {}

    const std::map<std::string, BenchValue>& getValues() const { return m_runData; }
    void addValue(const std::string& key, const BenchValue& value) { m_runData[key] = value; }

   private:
    std::map<std::string, BenchValue> m_runData;
};

/**
 * @brief Class for storing and managing benchmark results.
 *
 */
class Results {
   public:
    void RegisterRun(const RunData& runData) { m_runData.push_back(runData); }
    void SetMetadata(const std::map<std::string, BenchValue>& metadata) { m_metadata = metadata; }

    const std::vector<RunData>& getRuns() const { return m_runData; }
    const std::map<std::string, BenchValue>& getMetadata() const { return m_metadata; }

   private:
    std::vector<RunData> m_runData;
    std::map<std::string, BenchValue> m_metadata;
};
}  // namespace roccvbench
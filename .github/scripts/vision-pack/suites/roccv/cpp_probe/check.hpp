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

// Result lines for suites/roccv/harness/emit_checks.py:
//   @@VPCHECK<TAB>group<TAB>name<TAB>pass|fail|error<TAB>message
#pragma once

#include <cstdio>
#include <string>

inline void vp_check(const char* group, const std::string& name, const char* status, const std::string& msg) {
    std::string m = msg;
    for (auto& c : m)
        if (c == '\n' || c == '\t') c = ' ';
    std::printf("@@VPCHECK\t%s\t%s\t%s\t%s\n", group, name.c_str(), status, m.c_str());
    std::fflush(stdout);
}

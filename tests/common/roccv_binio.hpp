/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#pragma once

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <string>

namespace rocCVBinaryIO {
template <typename T>
bool write_array(std::string filename, const T *arr, size_t size) {
    if (!arr) {
        return false;
    }

    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if (!out.is_open()) {
        return false;
    }

    // header
    out << "rocCV Binary" << std::endl;
    out << "rocCV Array" << std::endl;

    // data
    out.write((char *)&size, sizeof(size_t));
    out.write((char *)arr, size * sizeof(T));
    out.close();

    return true;
}

template <typename T>
bool read_array(std::string filename, const T *arr, size_t size) {
    if (!arr) {
        return false;
    }

    std::ifstream in(filename, std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        return false;
    }

    // header
    std::string header;
    std::string type;

    std::getline(in, header);
    if (header != "rocCV Binary") {
        std::cout << "invalid binary file." << std::endl;
        return false;
    }
    std::getline(in, type);
    if (type != "rocCV Array") {
        std::cout << "invalid array file." << std::endl;
        return false;
    }

    // data
    size_t input_size = 0;
    in.read((char *)&input_size, sizeof(size_t));

    if (input_size != size) {
        std::cout << "input size is not the same size as given size."
                  << std::endl;
        return false;
    }

    in.read((char *)arr, size * sizeof(T));
    in.close();

    return true;
}

bool read_array_size(std::string filename, size_t &size) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);

    if (!in.is_open()) {
        return false;
    }

    // header
    std::string header;
    std::string type;
    std::string version;

    std::getline(in, header);
    if (header != "rocCV Binary") {
        std::cout << "invalid binary file." << std::endl;
        return false;
    }
    std::getline(in, type);
    if (type != "rocCV Array") {
        std::cout << "invalid array file." << std::endl;
        return false;
    }

    // data
    in.read((char *)&size, sizeof(size_t));
    in.close();

    return true;
}
};  // namespace rocCVBinaryIO
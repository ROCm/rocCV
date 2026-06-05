/**
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

#include "py_image_batch_var_shape.hpp"

#include <pybind11/stl.h>

#include <tuple>

PyImageBatchVarShape::PyImageBatchVarShape(int32_t capacity, eDeviceType device) {
    m_batch = std::make_shared<roccv::ImageBatchVarShape>(capacity, device);
}

std::shared_ptr<PyImageBatchVarShape> PyImageBatchVarShape::Create(int32_t capacity, eDeviceType device) {
    return std::make_shared<PyImageBatchVarShape>(capacity, device);
}

std::shared_ptr<PyImageBatchVarShape> PyImageBatchVarShape::WrapExternalBufferVector(std::vector<py::object> buffers,
                                                                                     roccv::ImageFormat format) {
    // Wrap each buffer first so the batch's device can be inferred from the
    // resulting images (they must all share a device).
    std::vector<std::shared_ptr<PyImage>> images;
    images.reserve(buffers.size());
    for (auto& buffer : buffers) {
        images.push_back(PyImage::WrapExternalBuffer(buffer, format));
    }

    eDeviceType device = images.empty() ? eDeviceType::GPU : images.front()->getDevice();
    auto batch = std::make_shared<PyImageBatchVarShape>(static_cast<int32_t>(images.size()), device);
    batch->pushBackMany(images);
    return batch;
}

void PyImageBatchVarShape::pushBack(std::shared_ptr<PyImage> image) {
    // Mutate the underlying batch first; if it throws (capacity/device mismatch)
    // the keep-alive mirror is left untouched.
    m_batch->pushBack(*image->getImage());
    m_images.push_back(image);
}

void PyImageBatchVarShape::pushBackMany(const std::vector<std::shared_ptr<PyImage>>& images) {
    const size_t oldSize = m_images.size();
    try {
        for (const auto& image : images) {
            m_batch->pushBack(*image->getImage());
            m_images.push_back(image);
        }
    } catch (...) {
        // Roll back to the pre-call state for the strong exception guarantee.
        const int32_t added = static_cast<int32_t>(m_images.size() - oldSize);
        if (added > 0) {
            m_batch->popBack(added);
            m_images.resize(oldSize);
        }
        throw;
    }
}

void PyImageBatchVarShape::popBack(int32_t count) {
    // popBack validates count before mutating, so the mirror stays consistent.
    m_batch->popBack(count);
    m_images.erase(m_images.end() - count, m_images.end());
}

void PyImageBatchVarShape::clear() {
    m_batch->clear();
    m_images.clear();
}

int32_t PyImageBatchVarShape::numImages() { return m_batch->numImages(); }

int32_t PyImageBatchVarShape::capacity() { return m_batch->capacity(); }

PySize2D PyImageBatchVarShape::maxSize() {
    roccv::Size2D size = m_batch->maxSize();
    return std::make_tuple(size.w, size.h);
}

roccv::ImageFormat PyImageBatchVarShape::uniqueFormat() { return m_batch->uniqueFormat(); }

eDeviceType PyImageBatchVarShape::getDevice() { return m_batch->device(); }

std::shared_ptr<PyImage> PyImageBatchVarShape::at(int32_t index) {
    // Support Python-style negative indexing.
    if (index < 0) {
        index += numImages();
    }
    if (index < 0 || index >= numImages()) {
        throw py::index_error("ImageBatchVarShape index out of range.");
    }
    return m_images[index];
}

const std::vector<std::shared_ptr<PyImage>>& PyImageBatchVarShape::images() const { return m_images; }

std::shared_ptr<roccv::ImageBatchVarShape> PyImageBatchVarShape::getBatch() { return m_batch; }

void PyImageBatchVarShape::Export(py::module& m) {
    using namespace py::literals;

    py::class_<PyImageBatchVarShape, std::shared_ptr<PyImageBatchVarShape>>(
        m, "ImageBatchVarShape", "A batch of variable-sized images that may differ in size, format, and dtype.")
        .def(py::init<int32_t, eDeviceType>(), "capacity"_a, "device"_a = eDeviceType::GPU,
             "Create an empty batch with the given capacity (maximum number of images).")
        .def("pushback", &PyImageBatchVarShape::pushBack, "image"_a, "Append a single image to the end of the batch.")
        .def("pushback", &PyImageBatchVarShape::pushBackMany, "images"_a,
             "Append multiple images to the end of the batch.")
        .def("popback", &PyImageBatchVarShape::popBack, "count"_a = 1,
             "Remove one or more images from the end of the batch.")
        .def("clear", &PyImageBatchVarShape::clear, "Remove all images from the batch; capacity is retained.")
        .def("__len__", &PyImageBatchVarShape::numImages, "Return the number of images currently in the batch.")
        .def("__getitem__", &PyImageBatchVarShape::at, "index"_a, "Return the image at the given index.")
        .def(
            "__iter__",
            [](PyImageBatchVarShape& self) { return py::make_iterator(self.images().begin(), self.images().end()); },
            py::keep_alive<0, 1>(), "Return an iterator over the images in the batch.")
        .def_property_readonly("capacity", &PyImageBatchVarShape::capacity,
                               "Read-only property returning the maximum number of images the batch can hold.")
        .def_property_readonly("maxsize", &PyImageBatchVarShape::maxSize,
                               "Read-only property returning the (width, height) bounding box across all images.")
        .def_property_readonly("uniqueformat", &PyImageBatchVarShape::uniqueFormat,
                               "Read-only property returning the common format, or Format.NONE if heterogeneous/empty.")
        .def_property_readonly("device", &PyImageBatchVarShape::getDevice,
                               "Read-only property returning the device the batch resides on.");

    m.def("as_images", &PyImageBatchVarShape::WrapExternalBufferVector, "buffers"_a, "format"_a = roccv::FMT_NONE,
          "Wraps a list of DLPack-supported buffers as a rocpycv ImageBatchVarShape without copying.");
}

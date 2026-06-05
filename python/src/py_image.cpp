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

#include "py_image.hpp"

#include <core/hip_assert.h>
#include <dlpack/dlpack.h>

#include <core/data_type.hpp>
#include <core/image_buffer.hpp>
#include <core/image_data.hpp>
#include <cstring>
#include <map>
#include <stdexcept>

#include "py_helpers.hpp"

namespace {

// Maps a (source, destination) device pair to the appropriate HIP memcpy kind.
const std::map<std::pair<eDeviceType, eDeviceType>, hipMemcpyKind> kDevicePairToMemcpyKind = {
    {{eDeviceType::CPU, eDeviceType::CPU}, hipMemcpyHostToHost},
    {{eDeviceType::CPU, eDeviceType::GPU}, hipMemcpyHostToDevice},
    {{eDeviceType::GPU, eDeviceType::CPU}, hipMemcpyDeviceToHost},
    {{eDeviceType::GPU, eDeviceType::GPU}, hipMemcpyDeviceToDevice}};

// Builds a DLManagedTensor describing an image as an interleaved HWC buffer:
// shape [height, width, channels], element-wise strides, single per-channel
// dtype. The owning PyImage is stashed in manager_ctx so the underlying buffer
// outlives every DLPack consumer.
DLManagedTensor* createDLManagedTensorFromImage(std::shared_ptr<roccv::Image> image, std::shared_ptr<PyImage> ctx) {
    auto imageData = image->exportData<roccv::ImageDataStrided>();
    const roccv::ImagePlaneStrided& plane = imageData.plane(0);

    const roccv::ImageFormat format = image->format();
    const int64_t channels = format.channels();
    const size_t elemSize = roccv::DataType(format.dtype()).size();

    DLManagedTensor* dlTensor = new DLManagedTensor();
    dlTensor->dl_tensor.data = plane.basePtr;
    dlTensor->dl_tensor.byte_offset = 0;
    dlTensor->dl_tensor.ndim = 3;
    dlTensor->dl_tensor.device = RoccvDeviceToDLDevice(image->device());
    dlTensor->dl_tensor.dtype = RoccvTypeToDLType(format.dtype());

    // HWC layout. DLPack strides are element-wise, so the row stride (bytes) is
    // converted to elements via the per-channel element size.
    dlTensor->dl_tensor.shape = new int64_t[3]{plane.height, plane.width, channels};
    dlTensor->dl_tensor.strides = new int64_t[3]{plane.rowStride / static_cast<int64_t>(elemSize), channels, 1};

    dlTensor->manager_ctx = new std::shared_ptr<PyImage>(ctx);
    dlTensor->deleter = [](DLManagedTensor* mt) {
        delete[] mt->dl_tensor.shape;
        delete[] mt->dl_tensor.strides;
        delete static_cast<std::shared_ptr<PyImage>*>(mt->manager_ctx);
        delete mt;
    };

    return dlTensor;
}

}  // namespace

PyImage::PyImage(PySize2D size, roccv::ImageFormat format, eDeviceType device) {
    auto [width, height] = size;
    m_image = std::make_shared<roccv::Image>(roccv::Size2D{width, height}, format, device);
}

PyImage::PyImage(std::shared_ptr<roccv::Image> image) : m_image(image) {}

PyImage::PyImage(std::shared_ptr<roccv::Image> image, DLManagedTensor* managedTensor)
    : m_image(image), m_managedTensor(managedTensor) {}

PyImage::~PyImage() {
    // If we are a consumer of a DLManagedTensor, ensure that we call its deleter.
    // The wrapped roccv::Image is view-only, so the buffer is released here.
    if (m_managedTensor && m_managedTensor->deleter) {
        m_managedTensor->deleter(m_managedTensor);
    }
}

std::shared_ptr<PyImage> PyImage::Zeros(PySize2D size, roccv::ImageFormat format, eDeviceType device) {
    auto [width, height] = size;
    auto image = std::make_shared<roccv::Image>(roccv::Size2D{width, height}, format, device);

    auto imageData = image->exportData<roccv::ImageDataStrided>();
    const roccv::ImagePlaneStrided& plane = imageData.plane(0);
    const size_t bytes = static_cast<size_t>(plane.rowStride) * plane.height;

    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipMemset(plane.basePtr, 0, bytes));
    } else {
        std::memset(plane.basePtr, 0, bytes);
    }

    return std::make_shared<PyImage>(image);
}

std::shared_ptr<PyImage> PyImage::WrapExternalBuffer(py::object src, roccv::ImageFormat format) {
    if (!py::hasattr(src, "__dlpack__")) {
        throw std::runtime_error("Provided object does not support the DLPack protocol.");
    }

    py::capsule dlpackCapsule = src.attr("__dlpack__")();
    if (!PyCapsule_IsValid(dlpackCapsule.ptr(), "dltensor")) {
        throw std::runtime_error("Invalid DLPack capsule.");
    }
    DLManagedTensor* dlManagedTensor = static_cast<DLManagedTensor*>(dlpackCapsule.get_pointer());
    DLTensor dlTensor = dlManagedTensor->dl_tensor;

    // Mark the capsule as consumed so its deleter will not free the underlying
    // data; ownership now flows through the returned PyImage's destructor.
    dlpackCapsule.set_name("used_dltensor");

    // Interpret the buffer as HWC (ndim 3) or HW grayscale (ndim 2).
    int64_t height, width, channels;
    if (dlTensor.ndim == 3) {
        height = dlTensor.shape[0];
        width = dlTensor.shape[1];
        channels = dlTensor.shape[2];
    } else if (dlTensor.ndim == 2) {
        height = dlTensor.shape[0];
        width = dlTensor.shape[1];
        channels = 1;
    } else {
        throw std::runtime_error("DLPack buffer must be 2-D (HW) or 3-D (HWC) to be wrapped as an image.");
    }

    // Infer the format from the buffer when one was not supplied.
    if (format == roccv::FMT_NONE) {
        format = roccv::ImageFormat(DLTypeToRoccvType(dlTensor.dtype), static_cast<int32_t>(channels));
    }

    const size_t elemSize = roccv::DataType(format.dtype()).size();

    // DLPack strides are element-wise; recover the byte row stride. A null stride
    // array denotes a compact (C-contiguous) buffer.
    int64_t rowStride;
    if (dlTensor.strides != nullptr) {
        rowStride = dlTensor.strides[0] * static_cast<int64_t>(elemSize);
    } else {
        rowStride = width * channels * static_cast<int64_t>(elemSize);
    }

    roccv::ImageBufferStrided strided{};
    strided.numPlanes = 1;
    strided.planes[0].width = static_cast<int32_t>(width);
    strided.planes[0].height = static_cast<int32_t>(height);
    strided.planes[0].rowStride = rowStride;
    strided.planes[0].basePtr = static_cast<char*>(dlTensor.data) + dlTensor.byte_offset;

    eDeviceType device = DLDeviceToRoccvDevice(dlTensor.device);

    // View-only wrap (no cleanup callback): the wrapped buffer is freed by this
    // PyImage's destructor via the consumed DLManagedTensor, mirroring PyTensor.
    roccv::Image image = (device == eDeviceType::GPU)
                             ? roccv::ImageWrapData(roccv::ImageDataStridedHip(format, strided))
                             : roccv::ImageWrapData(roccv::ImageDataStridedHost(format, strided));

    return std::make_shared<PyImage>(std::make_shared<roccv::Image>(image), dlManagedTensor);
}

py::capsule PyImage::toDLPack(py::object /* stream */) {
    // Stream parameter is intentionally unused to support framework DLPack device conversions.
    DLManagedTensor* dlTensor = createDLManagedTensorFromImage(m_image, shared_from_this());

    py::capsule capsule(dlTensor, "dltensor", [](PyObject* self) {
        if (PyCapsule_IsValid(self, "used_dltensor")) {
            return;  // Do nothing if the capsule has been consumed.
        }

        DLManagedTensor* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, "dltensor"));
        if (managed == nullptr) {
            PyErr_WriteUnraisable(self);
            return;
        }

        if (managed->deleter) {
            managed->deleter(managed);
        }
    });

    return capsule;
}

py::tuple PyImage::getDLDevice() {
    DLDevice device = RoccvDeviceToDLDevice(m_image->device());
    return py::make_tuple(py::int_(static_cast<int>(device.device_type)), py::int_(static_cast<int>(device.device_id)));
}

std::shared_ptr<PyImage> PyImage::copyTo(eDeviceType device) {
    auto dstImage = std::make_shared<roccv::Image>(m_image->size(), m_image->format(), device);

    auto srcData = m_image->exportData<roccv::ImageDataStrided>();
    auto dstData = dstImage->exportData<roccv::ImageDataStrided>();
    const roccv::ImagePlaneStrided& srcPlane = srcData.plane(0);
    const roccv::ImagePlaneStrided& dstPlane = dstData.plane(0);

    const roccv::ImageFormat format = m_image->format();
    const size_t elemSize = roccv::DataType(format.dtype()).size();
    const size_t widthBytes = static_cast<size_t>(srcPlane.width) * format.channels() * elemSize;

    // A pitched copy correctly handles differing source/destination row strides
    // (e.g. a wrapped buffer with arbitrary padding copied into a fresh image).
    HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(dstPlane.basePtr, dstPlane.rowStride, srcPlane.basePtr, srcPlane.rowStride,
                                       widthBytes, srcPlane.height,
                                       kDevicePairToMemcpyKind.at({m_image->device(), device})));

    return std::make_shared<PyImage>(dstImage);
}

PySize2D PyImage::getSize() {
    roccv::Size2D size = m_image->size();
    return std::make_tuple(size.w, size.h);
}

int PyImage::getWidth() { return m_image->size().w; }

int PyImage::getHeight() { return m_image->size().h; }

roccv::ImageFormat PyImage::getFormat() { return m_image->format(); }

eDeviceType PyImage::getDevice() { return m_image->device(); }

std::shared_ptr<roccv::Image> PyImage::getImage() { return m_image; }

void PyImage::Export(py::module& m) {
    using namespace py::literals;

    py::class_<PyImage, std::shared_ptr<PyImage>>(m, "Image",
                                                  "A single variable-sized image with device-resident data.")
        .def(py::init<PySize2D, roccv::ImageFormat, eDeviceType>(), "size"_a, "format"_a, "device"_a = eDeviceType::GPU,
             "Allocate a new image of the given size (width, height) and format.")
        .def_static("zeros", &PyImage::Zeros, "size"_a, "format"_a, "device"_a = eDeviceType::GPU,
                    "Create an image of the given size and format filled with zeros.")
        .def("copy_to", &PyImage::copyTo, "device"_a,
             "Returns a copy of the image with data copied to the given device.")
        .def("__dlpack__", &PyImage::toDLPack, "stream"_a = py::none(), "Creates a DLPack capsule from this image.")
        .def("__dlpack_device__", &PyImage::getDLDevice,
             "Returns a tuple containing the DLPack device type and id for this image.")
        .def_property_readonly("size", &PyImage::getSize,
                               "Read-only property returning the (width, height) of the image.")
        .def_property_readonly("width", &PyImage::getWidth, "Read-only property returning the width of the image.")
        .def_property_readonly("height", &PyImage::getHeight, "Read-only property returning the height of the image.")
        .def_property_readonly("format", &PyImage::getFormat, "Read-only property returning the format of the image.")
        .def_property_readonly("device", &PyImage::getDevice, "Read-only property returning the device of the image.");

    m.def("as_image", &PyImage::WrapExternalBuffer, "buffer"_a, "format"_a = roccv::FMT_NONE,
          "Wraps a DLPack-supported buffer as a rocpycv Image without copying.");
}

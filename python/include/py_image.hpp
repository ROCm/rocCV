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

#pragma once

#include <core/util_enums.h>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/image.hpp>
#include <core/image_format.hpp>
#include <memory>
#include <tuple>

namespace py = pybind11;

// Python-layer size type: a plain std::tuple<int, int> rather than a bound
// class. pybind11 auto-converts Python tuples/lists to it, and the binding
// converts to roccv::Size2D internally.
using PySize2D = std::tuple<int, int>;

/**
 * @brief Python container wrapping a single variable-sized roccv::Image.
 *
 * Mirrors the role PyTensor plays for roccv::Tensor: a refcounted handle that
 * keeps an underlying roccv::Image alive for the Python runtime and exposes its
 * metadata plus zero-copy interop. PyImage is the per-element type pushed into a
 * PyImageBatchVarShape.
 *
 * Two construction paths exist:
 *   - Allocating: build a fresh device buffer from (size, format).
 *   - Wrapping: adopt an externally-owned buffer exposed via DLPack (zero-copy),
 *     in which case lifetime is governed by the producer's DLManagedTensor
 *     deleter rather than by roccv::Image's own allocation.
 */
class PyImage : public std::enable_shared_from_this<PyImage> {
   public:
    /**
     * @brief Allocates a new image buffer of the given size and format on the
     * specified device, wrapping the resulting roccv::Image.
     *
     * @param size The image dimensions (width, height) in pixels.
     * @param format The pixel format (dtype + channel count + swizzle).
     * @param device The device the image buffer is allocated on.
     */
    PyImage(PySize2D size, roccv::ImageFormat format, eDeviceType device);

    /**
     * @brief Wraps an existing roccv::Image inside a newly constructed PyImage.
     *
     * @param image A shared pointer to the roccv::Image to wrap.
     */
    PyImage(std::shared_ptr<roccv::Image> image);

    /**
     * @brief Wraps a roccv::Image together with a DLManagedTensor consumed from
     * another framework. The underlying roccv::Image must reference the wrapped
     * buffer view-only; ownership is released back through the producer's
     * DLManagedTensor.deleter, invoked by this object's destructor.
     *
     * @param image A shared pointer to the roccv::Image to wrap.
     * @param managedTensor The DLManagedTensor provided by the producer.
     */
    PyImage(std::shared_ptr<roccv::Image> image, DLManagedTensor* managedTensor);

    /**
     * @brief Destroys the PyImage and invokes the wrapped DLManagedTensor's
     * deleter, if one exists and is valid.
     */
    ~PyImage();

    /**
     * @brief Creates a new image of the given size and format filled with zeros.
     *
     * @param size The image dimensions (width, height) in pixels.
     * @param format The pixel format.
     * @param device The device the image buffer is allocated on.
     * @return std::shared_ptr<PyImage>
     */
    static std::shared_ptr<PyImage> Zeros(PySize2D size, roccv::ImageFormat format, eDeviceType device);

    /**
     * @brief Wraps an external buffer (a capsule containing a DLManagedTensor)
     * as an image without copying.
     *
     * @param src A capsule containing a DLManagedTensor.
     * @param format The pixel format to interpret the buffer as.
     * @return std::shared_ptr<PyImage>
     */
    static std::shared_ptr<PyImage> WrapExternalBuffer(py::object src, roccv::ImageFormat format);

    /**
     * @brief Exports this image as a capsule containing a DLManagedTensor for
     * consumption by another framework.
     *
     * @param stream Optional stream pointer value (used for framework interop).
     * @return py::capsule
     */
    py::capsule toDLPack(py::object stream);

    /**
     * @brief Returns a tuple describing the DLPack device of this image. Index 0
     * is the device type and index 1 is the device id.
     *
     * @return py::tuple
     */
    py::tuple getDLDevice();

    /**
     * @brief Creates a copy of this image on the specified device.
     *
     * @param device The device of the new image.
     * @return std::shared_ptr<PyImage>
     */
    std::shared_ptr<PyImage> copyTo(eDeviceType device);

    /**
     * @brief Gets the size (width, height) of this image.
     *
     * @return PySize2D
     */
    PySize2D getSize();

    /**
     * @brief Gets the width of this image in pixels.
     *
     * @return int
     */
    int getWidth();

    /**
     * @brief Gets the height of this image in pixels.
     *
     * @return int
     */
    int getHeight();

    /**
     * @brief Gets the pixel format of this image.
     *
     * @return roccv::ImageFormat
     */
    roccv::ImageFormat getFormat();

    /**
     * @brief Gets the device this image resides on.
     *
     * @return eDeviceType
     */
    eDeviceType getDevice();

    /**
     * @brief Gets the underlying roccv::Image this container wraps.
     *
     * @return std::shared_ptr<roccv::Image>
     */
    std::shared_ptr<roccv::Image> getImage();

    /**
     * @brief Exports this class in the provided module.
     *
     * @param m The python module to export this class to.
     */
    static void Export(py::module& m);

   private:
    std::shared_ptr<roccv::Image> m_image;

    // TODO: DLManagedTensor is slated for deprecation in future DLPack versions.
    // Update this to the versioned managed tensor before then.
    DLManagedTensor* m_managedTensor = nullptr;
};

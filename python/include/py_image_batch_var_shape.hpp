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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/image_batch_var_shape.hpp>
#include <core/image_format.hpp>
#include <memory>
#include <vector>

#include "py_image.hpp"

namespace py = pybind11;

/**
 * @brief Python container wrapping a roccv::ImageBatchVarShape.
 *
 * Holds a batch of variable-sized images that may differ in size, format, and
 * dtype. Capacity is fixed at construction; images are appended via pushBack up
 * to that capacity.
 *
 * The underlying roccv::ImageBatchVarShape is move-only and stores its images by
 * value, so this wrapper keeps a parallel vector of PyImage handles. That mirror
 * serves two purposes: it keeps each image's Python object (and any wrapped
 * DLManagedTensor) alive for as long as the batch references it, and it backs
 * Python-level indexing/iteration without rebuilding handles from the underlying
 * batch.
 */
class PyImageBatchVarShape : public std::enable_shared_from_this<PyImageBatchVarShape> {
   public:
    /**
     * @brief Constructs an empty batch with the given capacity on the specified
     * device, wrapping a newly constructed roccv::ImageBatchVarShape.
     *
     * @param capacity The maximum number of images the batch can hold.
     * @param device The device the batch (and every image it accepts) resides on.
     */
    PyImageBatchVarShape(int32_t capacity, eDeviceType device);

    ~PyImageBatchVarShape() = default;

    /**
     * @brief Creates a new batch with the given capacity on the specified device.
     *
     * @param capacity The maximum number of images the batch can hold.
     * @param device The device the batch resides on.
     * @return std::shared_ptr<PyImageBatchVarShape>
     */
    static std::shared_ptr<PyImageBatchVarShape> Create(int32_t capacity, eDeviceType device);

    /**
     * @brief Builds a batch by wrapping a vector of external buffers (capsules
     * containing DLManagedTensors) as images without copying. Capacity is set to
     * the number of buffers provided.
     *
     * @param buffers A list of capsules, each containing a DLManagedTensor.
     * @param format The pixel format to interpret each buffer as.
     * @return std::shared_ptr<PyImageBatchVarShape>
     */
    static std::shared_ptr<PyImageBatchVarShape> WrapExternalBufferVector(std::vector<py::object> buffers,
                                                                          roccv::ImageFormat format);

    /**
     * @brief Appends a single image to the end of the batch. Throws if capacity
     * would be exceeded or the image's device does not match the batch's.
     *
     * @param image The image to append.
     */
    void pushBack(std::shared_ptr<PyImage> image);

    /**
     * @brief Appends multiple images to the end of the batch. Provides a strong
     * exception guarantee: on failure the batch is rolled back to its pre-call
     * state.
     *
     * @param images The images to append.
     */
    void pushBackMany(const std::vector<std::shared_ptr<PyImage>>& images);

    /**
     * @brief Removes the trailing `count` images from the batch.
     *
     * @param count The number of images to remove.
     */
    void popBack(int32_t count);

    /**
     * @brief Removes all images from the batch. Capacity is retained and the
     * batch remains reusable.
     */
    void clear();

    /**
     * @brief Gets the number of images currently in the batch (__len__).
     *
     * @return int32_t
     */
    int32_t numImages();

    /**
     * @brief Gets the maximum number of images the batch can hold.
     *
     * @return int32_t
     */
    int32_t capacity();

    /**
     * @brief Gets the bounding box (width, height) across all images in pixels.
     *
     * @return PySize2D
     */
    PySize2D maxSize();

    /**
     * @brief Gets the common format across all images, or FMT_NONE if the formats
     * are heterogeneous or the batch is empty.
     *
     * @return roccv::ImageFormat
     */
    roccv::ImageFormat uniqueFormat();

    /**
     * @brief Gets the device this batch resides on.
     *
     * @return eDeviceType
     */
    eDeviceType getDevice();

    /**
     * @brief Gets the image at the given index (__getitem__).
     *
     * @param index The index of the image to retrieve.
     * @return std::shared_ptr<PyImage>
     */
    std::shared_ptr<PyImage> at(int32_t index);

    /**
     * @brief Gets the list of PyImage handles currently held by the batch. Backs
     * Python-level iteration over the batch.
     *
     * @return const std::vector<std::shared_ptr<PyImage>>&
     */
    const std::vector<std::shared_ptr<PyImage>>& images() const;

    /**
     * @brief Gets the underlying roccv::ImageBatchVarShape this container wraps.
     *
     * @return std::shared_ptr<roccv::ImageBatchVarShape>
     */
    std::shared_ptr<roccv::ImageBatchVarShape> getBatch();

    /**
     * @brief Exports this class in the provided module.
     *
     * @param m The python module to export this class to.
     */
    static void Export(py::module& m);

   private:
    std::shared_ptr<roccv::ImageBatchVarShape> m_batch;

    // Keep-alive mirror of the images pushed into m_batch. Keeps each PyImage
    // (and any wrapped DLManagedTensor) alive for the batch's lifetime and backs
    // Python indexing/iteration.
    std::vector<std::shared_ptr<PyImage>> m_images;
};

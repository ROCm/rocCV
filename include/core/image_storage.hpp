/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
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

namespace roccv {

/**
 * @brief Holds the raw data pointer for a single Image and serves as the
 * refcount target shared between Image handles.
 *
 * ImageStorage carries no lifecycle logic of its own: freeing the underlying
 * buffer is the responsibility of the shared_ptr<ImageStorage> deleter
 * installed at the Image construction site. The allocating Image ctor
 * captures the allocator + device into its deleter; ImageWrapData captures
 * the user's cleanup callback (or installs none for the view-only case).
 *
 * As a result, ImageStorage is held only by shared_ptr — never by value, never
 * copied. Move/copy are deleted to enforce that.
 */
class ImageStorage {
   public:
    explicit ImageStorage(void* data) : m_data(data) {}

    ImageStorage(const ImageStorage&) = delete;
    ImageStorage& operator=(const ImageStorage&) = delete;

    /**
     * @brief Returns the raw data pointer this storage is tracking.
     */
    void* data() const noexcept { return m_data; }

   private:
    void* m_data;
};

}  // namespace roccv

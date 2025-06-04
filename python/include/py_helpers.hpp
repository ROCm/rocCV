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

#include <core/util_enums.h>
#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Converts a DLDataType to a rocCV data type.
 *
 * @param dtype The DLDataType
 * @return eDataType
 */
extern eDataType DLTypeToRoccvType(DLDataType dtype);

/**
 * @brief Converts a DLDevice to a rocCV device.
 *
 * @param device The DLDevice.
 * @return eDeviceType
 */
extern eDeviceType DLDeviceToRoccvDevice(DLDevice device);

/**
 * @brief Converts a rocCV device to a DLDevice. Currently, this will set the device id to 0 if the device is the CPU,
 * and 1 if the device is a GPU. This needs to be updated later, as rocCV should support setting the device id
 * appropriately.
 *
 * @param device The rocCV device.
 * @return DLDevice
 */
extern DLDevice RoccvDeviceToDLDevice(eDeviceType device);

/**
 * @brief Converts a rocCV data type to a DLDataType.
 *
 * @param dtype The rocCV data type.
 * @return DLDataType
 */
extern DLDataType RoccvTypeToDLType(eDataType dtype);

/**
 * @brief Creates a float4 given a python list. This also ensures that the provided list is of size 4 and will throw a
 * runtime error otherwise.
 *
 * @param src A python list of size 4.
 * @return float4
 */
extern float4 GetFloat4FromPyList(py::list src);

/**
 * @brief Creates a double2 given a python tuple. Ensures that the provided tuple is of the right size.
 *
 * @param src A python tuple of size 2.
 * @return double2
 */
extern double2 GetDouble2FromTuple(py::tuple src);
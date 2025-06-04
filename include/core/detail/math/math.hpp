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

#include <hip/hip_runtime.h>

#include <cassert>

namespace roccv::detail::math {

/**
 * @brief Defines a Vector object.
 *
 * @tparam T The underlying data type of the vector.
 * @tparam N The size of the vector.
 */
template <class T, int N>
class Vector {
   public:
    using Type = T;

    /**
     * @brief Returns the size of the vector.
     *
     * @return The size of the vector.
     */
    __host__ __device__ int size() const { return N; }

    const __host__ __device__ T &operator[](int i) const {
        assert(i >= 0 && i < size());
        return data_[i];
    }

    __host__ __device__ T &operator[](int i) {
        assert(i >= 0 && i < size());
        return data_[i];
    }

    __host__ __device__ operator const T *() const { return &data_[0]; }

    __host__ __device__ operator T *() { return &data_[0]; }

    T data_[N];
};

template <class T, int N>
__host__ __device__ Vector<T, N> operator-(const Vector<T, N> &v) {
    Vector<T, N> r;
    for (int i = 0; i < N; ++i) {
        r[i] = -v[i];
    }
    return r;
}

/**
 * @brief Defines a matrix object.
 *
 * @tparam T The type of the matrix.
 * @tparam M Number of rows in the matrix.
 * @tparam N Number of columns in the matrix.
 */
template <class T, int M, int N = M>
class Matrix {
   public:
    using Type = T;

    /**
     * @brief Returns the number of rows in a matrix.
     *
     * @return The number of rows in the matrix.
     */
    __host__ __device__ int rows() const { return M; }

    /**
     * @brief Returns the number of columns in the matrix.
     *
     * @return The number of columns in the matrix.
     */
    __host__ __device__ int cols() const { return N; }

    /**
     * @brief Returns a row of a matrix as a constant vector reference.
     *
     * @param i The row to return.
     * @return A row of the matrix represented as a vector.
     */
    const __host__ __device__ Vector<T, N> &operator[](int i) const {
        assert(i >= 0 && i < rows());
        return data_[i];
    }

    /**
     * @brief Returns a row of a matrix as a vector reference.
     *
     * @param i The row of the matrix to return.
     * @return A reference to a vector representing a row of the matrix.
     */
    __host__ __device__ Vector<T, N> &operator[](int i) {
        assert(i >= 0 && i < rows());
        return data_[i];
    }

    /**
     * @brief Returns a constant data reference from the matrix.
     *
     * @param c An int2 coordinate in the form {col, row}
     * @return A constant data reference from the matrix.
     */
    const __host__ __device__ T &operator[](int2 c) const {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return data_[c.y][c.x];
    }

    /**
     * @brief Returns a data reference from the matrix.
     *
     * @param c An int2 coordinate in the form {col, row}
     * @return A data reference from the matrix.
     */
    __host__ __device__ T &operator[](int2 c) {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return data_[c.y][c.x];
    }

    /**
     * @brief Loads a row-major array into the matrix.
     *
     * @param data An array of type T.
     */
    __host__ __device__ void load(const T *data) {
#pragma unroll
        for (int row = 0; row < M; row++) {
#pragma unroll
            for (int col = 0; col < N; col++) {
                data_[row][col] = data[row * M + col];
            }
        }
    }

    /**
     * @brief Stores data from the matrix into a 1D array in row-major order.
     *
     * @param data The array to store matrix data in.
     */
    __host__ __device__ void store(T *data) {
#pragma unroll
        for (int row = 0; row < M; row++) {
#pragma unroll
            for (int col = 0; col < N; col++) {
                data[row * M + col] = data_[row][col];
            }
        }
    }

    Vector<T, N> data_[M];
};

// Determinant calculations
template <class T>
constexpr __host__ __device__ T det(const Matrix<T, 0, 0> &m) {
    return T{1};
}

template <class T>
constexpr __host__ __device__ T det(const Matrix<T, 1, 1> &m) {
    return m[0][0];
}

template <class T>
constexpr __host__ __device__ T det(const Matrix<T, 2, 2> &m) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template <class T>
constexpr __host__ __device__ T det(const Matrix<T, 3, 3> &m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Matrix inversion
template <class T>
constexpr __host__ __device__ Matrix<T, 3, 3> inv(const Matrix<T, 3, 3> &m, const T &d) {
    Matrix<T, 3, 3> A;
    A[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    A[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
    A[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    A[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
    A[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    A[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
    A[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    A[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
    A[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;

    return A;
}

template <class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 3, 3> &m, const T &d) {
    m = inv(m, d);
}

template <class T, int N, class = std::enable_if_t<(N < 4)>>
constexpr __host__ __device__ bool inv_inplace(Matrix<T, N, N> &m) {
    T d = det(m);

    if (d == 0) {
        return false;
    }

    inv_inplace(m, d);

    return true;
}

template <class T>
__host__ __device__ void swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

template <class T, int N>
__host__ __device__ T dot(const Vector<T, N> &a, const Vector<T, N> &b) {
    T d = a[0] * b[0];
#pragma unroll
    for (int j = 1; j < a.size(); ++j) {
        d += a[j] * b[j];
    }
    return d;
}

template <class F = float, class T, int N>
__host__ __device__ bool lu_inplace(Matrix<T, N, N> &m, Vector<int, N> &p) {
    Vector<F, N> v;

#pragma unroll
    for (int i = 0; i < N; ++i) {
        F big = 0;

#pragma unroll
        for (int j = 0; j < N; ++j) {
            big = max(big, abs(m[i][j]));
        }

        if (big == 0) {
            return false;
        }

        v[i] = 1.0 / big;
    }

#pragma unroll
    for (int k = 0; k < N; ++k) {
        F big = 0;
        int imax = k;

#pragma unroll
        for (int i = k; i < N; ++i) {
            F aux = v[i] * abs(m[i][k]);

            if (aux > big) {
                big = aux;
                imax = i;
            }
        }

        if (k != imax) {
            swap(m[imax], m[k]);

            v[imax] = v[k];
        }

        p[k] = imax;

        if (m[k][k] == 0) {
            return false;
        }

#pragma unroll
        for (int i = k + 1; i < N; ++i) {
            T aux = m[i][k] /= m[k][k];

#pragma unroll
            for (int j = k + 1; j < N; ++j) {
                m[i][j] -= aux * m[k][j];
            }
        }
    }

    return true;
}

template <class T, int N>
__host__ __device__ void solve_inplace(const Matrix<T, N, N> &lu, const Vector<int, N> &p, Vector<T, N> &b) {
    int ii = -1;

#pragma unroll
    for (int i = 0; i < N; ++i) {
        int ip = p[i];
        T sum = b[ip];
        b[ip] = b[i];

        if (ii >= 0) {
            for (int j = ii; j < i; ++j) {
                sum -= lu[i][j] * b[j];
            }
        } else if (sum != 0) {
            ii = i;
        }

        b[i] = sum;
    }

#pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        T sum = b[i];

#pragma unroll
        for (int j = i + 1; j < N; ++j) {
            sum -= lu[i][j] * b[j];
        }

        b[i] = sum / lu[i][i];
    }
}

template <class T, int N>
__host__ __device__ bool solve_inplace(const Matrix<T, N, N> &m, Vector<T, N> &b) {
    Vector<int, N> p;
    Matrix<T, N, N> LU = m;

    if (!lu_inplace(LU, p)) {
        return false;
    }

    solve_inplace(LU, p, b);

    return true;
}

};  // namespace roccv::detail::math

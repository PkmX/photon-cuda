#ifndef RGBA_HPP
#define RGBA_HPP

#include <array>
#include "rgb.hpp"

template<typename T>
class rgba {
    public:
        T c[4];

        __host__ __device__ rgba() {}
        __host__ __device__ rgba(rgb<T> rhs) : c{rhs[0], rhs[1], rhs[2], 1} {}
        __host__ __device__ rgba(T r, T g, T b, T a) : c{r, g, b, a} {}

        __host__ __device__ T& operator[](const std::size_t n) { return c[n]; }
        __host__ __device__ const T& operator[](const std::size_t n) const { return c[n]; }
};

using rgbaf = rgba<float>;

#endif // RGBA_HPP

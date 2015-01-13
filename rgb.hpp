#ifndef RGB_HPP
#define RGB_HPP

#include <array>
#include <boost/algorithm/clamp.hpp>
using boost::algorithm::clamp;

template<typename T>
struct rgb {
    T c[3];

    __host__ __device__ rgb() {}
    __host__ __device__ rgb(T r, T g, T b) : c{r, g, b} {}

    __host__ __device__ T& operator[](const std::size_t n) { return c[n]; }
    __host__ __device__ const T& operator[](const std::size_t n) const { return c[n]; }

    static const rgb<T> black;
};

using rgbf = rgb<float>;

template<typename T>
const rgb<T> rgb<T>::black = rgb<T>{0, 0, 0};

template<typename T>
__host__ __device__ rgb<T> operator+(const rgb<T> c1, const rgb<T> c2) {
    return {c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]};
}

template<typename T>
__host__ __device__ rgb<T> operator*(const rgb<T> c1, const rgb<T> c2) {
    return {c1[0] * c2[0], c1[1] * c2[1], c1[2] * c2[2]};
}

template<typename T>
__host__ __device__ rgb<T>& operator+=(rgb<T>& c1, const rgb<T> c2) {
    return (c1 = c1 + c2);
}

template<typename T>
__host__ __device__ rgb<T>& operator*=(rgb<T>& c1, const rgb<T> c2) {
    return (c1 = c1 * c2);
}

template<typename T>
__host__ __device__ rgb<T> operator*(const rgb<T> c, const T t) {
    return {c[0] * t, c[1] * t, c[2] * t};
}

template<typename T>
__host__ __device__ rgb<T> operator/(const rgb<T> c, const T t) {
    return {c[0] / t, c[1] / t, c[2] / t};
}

template<typename T>
__host__ __device__ rgb<T> clamp(const rgb<T> c, const T lo, const T hi) {
#ifndef __CUDA_ARCH__
    return {clamp(c[0], lo, hi), clamp(c[1], lo, hi), clamp(c[2], lo, hi)};
#else
    return { (c[0] <= lo) ? lo : (c[0] >= hi ? hi : c[0]), (c[1] <= lo) ? lo : (c[1] >= hi ? hi : c[1]), (c[2] <= lo) ? lo : (c[2] >= hi ? hi : c[2]) };
#endif
}

#endif // RGB_HPP

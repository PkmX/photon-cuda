#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <cmath>
#include <array>

template<typename T>
struct vector3 {
    T v[3];

    __host__ __device__ vector3() {}
    __host__ __device__ vector3(T x, T y, T z): v {x, y, z} {}

    __host__ __device__ T& operator[](const std::size_t n) { return v[n]; }
    __host__ __device__ const T& operator[](const std::size_t n) const { return v[n]; }
};

using vector3f = vector3<float>;

template<typename T>
__host__ __device__ auto operator-(const vector3<T> v) -> vector3<T> {
    return {-v[0], -v[1], -v[2]};
}

template<typename T>
__host__ __device__ auto operator-(const vector3<T> v1, const vector3<T> v2) -> vector3<T> {
    return {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
}

template<typename T>
__host__ __device__ vector3<T> operator*(const vector3<T> v, const T t)
{
    return {v[0] * t, v[1] * t, v[2] * t};
}

template<typename T>
__host__ __device__ vector3<T> operator/(const vector3<T> v, const T t)
{
    return {v[0] / t, v[1] / t, v[2] / t};
}

template<typename T>
__host__ __device__ T dot(const vector3<T> v1, const vector3<T> v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename T>
__host__ __device__ auto cross(const vector3<T> v1, const vector3<T> v2) -> vector3<T> {
    return {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
}

template<typename T>
__host__ __device__ auto scalar_proj(const vector3<T> v1, const vector3<T> v2) -> T {
    return dot(v1, v2) / length(v2);
}

template<typename T>
__host__ __device__ T length2(const vector3<T> v)
{
    return dot(v, v);
}

template<typename T>
__host__ __device__ T length(const vector3<T> v)
{
    return std::sqrt(length2(v));
}

#endif // VECTOR3_HPP

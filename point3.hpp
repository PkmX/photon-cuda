#ifndef POINT3_HPP
#define POINT3_HPP

#include <host_defines.h>
#include "vector3.hpp"

template<typename T>
class point3
{
    public:
        T p[3];

        __host__ __device__ point3() {}
        __host__ __device__ point3(T x, T y, T z) : p{x, y, z} {}

        __host__ __device__ T& operator[](const std::size_t n) { return p[n]; }
        __host__ __device__ const T& operator[](const std::size_t n) const { return p[n]; }
};

typedef point3<float> point3f;
typedef point3<double> point3d;

template<typename T>
__host__ __device__ point3<T> operator+(const point3<T>& p, const vector3<T>& v)
{
    return {p[0] + v[0], p[1] + v[1], p[2] + v[2]};
}

template<typename T>
__host__ __device__ vector3<T> operator-(const point3<T>& p1, const point3<T>& p2)
{
    return {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
}

template<typename T>
__host__ __device__ auto distance(const point3<T> p1, const point3<T> p2) -> T {
    return length2(p2 - p1);
}

#endif // POINT3_HPP

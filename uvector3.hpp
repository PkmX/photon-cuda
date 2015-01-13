#ifndef UVECTOR3_HPP
#define UVECTOR3_HPP

#include <array>
#include "vector3.hpp"

template<typename T>
class uvector3
{
    public:
        __host__ __device__ explicit uvector3(const vector3<T> v) : u{v[0] / length(v), v[1] / length(v), v[2] / length(v)} {}

        __host__ __device__ operator vector3<T>() const { return {u[0], u[1], u[2]}; }

        __host__ __device__ const T& operator[](const std::size_t n) const { return u[n]; }

    private:
        T u[3];
};

using uvector3f = uvector3<float>;

template<typename T>
__host__ __device__ uvector3<T> normalize(const vector3<T> v)
{
    return uvector3<T>(v);
}

template<typename T>
__host__ __device__ uvector3<T> operator-(const uvector3<T> u)
{
    return uvector3<T>{vector3<T>{-u[0], -u[1], -u[2]}};
}

template<typename T>
__host__ __device__ vector3<T> operator-(const vector3<T> v1, const uvector3<T> v2) {
    return v1 - static_cast<const vector3<T>>(v2);
}

template<typename T>
__host__ __device__ vector3<T> operator*(const T t, const uvector3<T> v)
{
    return static_cast<const vector3<T>>(v) * t;
}

template<typename T>
__host__ __device__ vector3<T> operator*(const uvector3<T> v, const T t)
{
    return static_cast<const vector3<T>>(v) * t;
}

template<typename T>
__host__ __device__ T dot(const uvector3<T> u, const vector3<T> v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

template<typename T>
__host__ __device__ T dot(const uvector3<T> u1, const uvector3<T> u2)
{
    return u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];
}

#endif // UVECTOR3_HPP

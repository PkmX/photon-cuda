#ifndef LINE3_HPP
#define LINE3_HPP

#include "point3.hpp"
#include "vector3.hpp"
#include "uvector3.hpp"

template<typename T>
class line3
{
    public:
        point3<T> p;
        vector3<T> v;

        __host__ __device__ point3<T> operator()(T t) const { return p + t * v; }
};

using line3f = line3<float>;

template<typename T>
class uline3
{
    public:
        point3<T> p;
        uvector3<T> v;

        __host__ __device__ point3<T> operator()(T t) const { return p + t * v; }
};

using uline3f = uline3<float>;

#endif // LINE3_HPP

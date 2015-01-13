#ifndef PARALLELOGRAM_HPP
#define PARALLELOGRAM_HPP

#include "point3.hpp"
#include "vector3.hpp"

template<typename T>
class parallelogram
{
    public:
        point3<T> p;
        vector3<T> vx;
        vector3<T> vy;

        __host__ __device__ point3<T> operator()(T x, T y) const { return p + vx * x + vy * y; }
};

#endif // PARALLELOGRAM_HPP

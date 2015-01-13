#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "point3.hpp"

template<typename T>
class sphere
{
    public:
        point3<T> c;
        T r;
};

#endif // SPHERE_HPP

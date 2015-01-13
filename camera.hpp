#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "point3.hpp"
#include "parallelogram.hpp"

template<typename T>
class camera_type
{
    public:
        point3<T> position;
        parallelogram<T> image_plane;
};

#endif // CAMERA_HPP

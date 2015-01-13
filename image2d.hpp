#ifndef IMAGE2D_HPP
#define IMAGE2D_HPP

#include <vector>
#include "rgba.hpp"

template<typename T>
class image2d
{
    private:
        typedef std::vector<T> container_type;

    public:
        typedef typename container_type::value_type value_type;
        typedef typename container_type::size_type size_type;
        typedef typename container_type::iterator iterator;
        typedef typename container_type::const_iterator const_iterator;

        image2d(const size_type w, const size_type h);

        T& operator()(const size_type x, const size_type y);
        const T& operator()(const size_type x, const size_type y) const;

        iterator begin();
        const_iterator begin() const;

        iterator end();
        const_iterator end() const;

        size_type width() const;
        size_type height() const;

    private:
        container_type data;
        size_type width_;
        size_type height_;
};

template<typename T>
image2d<T>::image2d(const size_type w, const size_type h)
    : data(w * h),
      width_(w),
      height_(h)
{
}

template<typename T>
T& image2d<T>::operator()(const size_type x, const size_type y)
{
    return data[x + width() * y];
}

template<typename T>
const T& image2d<T>::operator()(const size_type x, const size_type y) const
{
    return data[x + width() * y];
}

template<typename T>
typename image2d<T>::iterator image2d<T>::begin()
{
    return data.begin();
}

template<typename T>
typename image2d<T>::const_iterator image2d<T>::begin() const
{
    return data.begin();
}

template<typename T>
typename image2d<T>::iterator image2d<T>::end()
{
    return data.end();
}

template<typename T>
typename image2d<T>::const_iterator image2d<T>::end() const
{
    return data.end();
}

template<typename T>
typename image2d<T>::size_type image2d<T>::width() const
{
    return width_;
}

template<typename T>
typename image2d<T>::size_type image2d<T>::height() const
{
    return height_;
}

template<typename T>
void clear(image2d<rgba<T>>& img)
{
    for (auto& c: img)
    {
        c[0] = 0;
        c[1] = 0;
        c[2] = 0;
        c[3] = std::numeric_limits<T>::max();
    }
}

#endif // IMAGE2D_HPP

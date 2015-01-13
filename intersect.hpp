#ifndef INTERSECT_HPP
#define INTERSECT_HPP

#include <tuple>
#include <boost/optional.hpp>
#include "sphere.hpp"
#include "line3.hpp"

template<typename T>
auto line_intersect(const uline3<T>& r, const sphere<T>& s) -> boost::optional<std::tuple<T, T>>
{
    const auto c = s.c - r.p;
    const auto b = dot(r.v, c);
    const auto d = s.r * s.r - (length2(c) - b * b);

    if (d >= 0) {
        return std::tuple<T, T>{b - std::sqrt(d), b + std::sqrt(d)};
    } else {
        return {};
    }
}

template<typename T>
auto ray_intersect(const uline3<T>& r, const parallelogram<T>& p) -> boost::optional<T> {
    const auto n = [&]() {
        const auto n = cross(p.vx, p.vy);
        return normalize(dot(n, p.p - r.p) < 0 ? n : -n);
    }();
    const auto vn = dot(r.v, n);

    if (vn < 0) {
        const auto t = dot(p.p - r.p, static_cast<vector3<T>>(n)) / dot(r.v, static_cast<vector3<T>>(n));
        const auto rt = r(t);
        const auto sx = scalar_proj(rt - p.p, p.vx);
        const auto sy = scalar_proj(rt - p.p, p.vy);

        if (0 <= sx && sx <= length(p.vx) && 0 <= sy && sy <= length(p.vy)) {
            return boost::optional<T>{t};
        } else {
            return {};
        }
    } else {
        return {};
    }
}

template<typename T>
auto ray_intersect(const uline3<T>& r, const sphere<T>& t) -> boost::optional<T> {
    const auto intersect_result = line_intersect(r, t);
    if (intersect_result) {
        const auto ts = intersect_result;
        const auto t0 = std::get<0>(*ts), t1 = std::get<1>(*ts);

        if (t0 > 0 && t0 <= t1) {
            return boost::optional<T>(t0);
        } else if (t1 > 0) {
            return boost::optional<T>(t1);
        }
    }

    return boost::optional<T>();
}

template<typename T, typename InputIterator>
auto ray_intersect(const uline3<T>& r, InputIterator first, InputIterator last) -> std::tuple<InputIterator, T>
{
    std::tuple<InputIterator, T> result{last, std::numeric_limits<T>::max()};

    while (first != last) {
        const auto t = ray_intersect(r, first->mesh);
        if (t && *t > 0 && *t < std::get<1>(result)) {
            std::get<0>(result) = first;
            std::get<1>(result) = *t;
        }

        ++first;
    }

    return result;
}

#endif // INTERSECT_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>
#include <boost/log/trivial.hpp>
#include <boost/optional.hpp>

#include "camera.hpp"
#include "image2d.hpp"
#include "intersect.hpp"
#include "line3.hpp"
#include "rgb.hpp"
#include "sphere.hpp"
#include "tga.hpp"
#include "uvector3.hpp"

struct point_light_type {
    point3f position;
    rgb<float> color;
    size_t num_photons;
};

struct phong_type {
    rgb<float> diffuse_color, specular_color;
    float shininess;

    auto operator()(const uvector3f& wi, const uvector3f& wo, const uvector3f& n) const -> rgb<float> {
        const auto nl = dot(n, wi);
        const auto r = normalize(2 * dot(wi, n) * n - wi);
        const auto re = dot(r, wo);

        return (nl > 0.0f ? diffuse_color * nl : rgbf::black) + (re > 0.0f ? specular_color * std::pow(re, shininess) : rgbf::black);
    };
};

struct material_type {
    float diffuse_probability, specular_probability;
    phong_type brdf;
};

struct sphere_object_type {
    sphere<float> mesh;
    material_type material;
};

struct parallelogram_object_type {
    parallelogram<float> mesh;
    material_type material;
};

struct world_type {
    std::vector<sphere_object_type> sphere_objects;
    std::vector<parallelogram_object_type> parallelogram_objects;
    std::vector<point_light_type> point_lights;
};

struct photon_type {
    point3f position;
    uvector3f direction;
    rgb<float> color;
    std::uint8_t depth;
};

class kdtree {
    public:
        kdtree(std::vector<photon_type>&& src) : v(std::move(src)) { build(v.begin(), v.end(), 0); }

        auto knn(const point3f p, const std::size_t k) const -> std::vector<photon_type> {
            // return knn_impl(p, k, std::numeric_limits<float>::infinity(), v.begin(), v.end(), 0);
            std::vector<photon_type> result;
            knn_impl(p, k, v.begin(), v.end(), 0, result);
            return result;
        }

        auto vector() const -> const std::vector<photon_type>& { return v; }

    private:
        static auto knn_impl(const point3f p, const std::size_t k, const std::vector<photon_type>::const_iterator first, const std::vector<photon_type>::const_iterator last, const std::size_t axis, std::vector<photon_type>& out) -> void {
            if (last - first > 1) {
                const auto median = first + (last - first) / 2;
                const auto nearer_range = p[axis] < median->position[axis] ? std::make_tuple(first, median) : std::make_tuple(median, last);
                const auto farther_range = p[axis] < median->position[axis] ? std::make_tuple(median, last) : std::make_tuple(first, median);
                const auto next_axis = (axis + 1) % 3;

                knn_impl(p, k, std::get<0>(nearer_range), std::get<1>(nearer_range), next_axis, out);
                if (out.size() < k || (out.size() > 0 ? distance(out.back().position, p) >= std::abs(median->position[axis] - p[axis]) : true)) {
                    knn_impl(p, k, std::get<0>(farther_range), std::get<1>(farther_range), next_axis, out);
                }
            } else if (last - first == 1) {
                const auto insert_it = std::upper_bound(out.begin(), out.end(), *first, [p](const auto p1, const auto p2) { return distance(p1.position, p) < distance(p2.position, p); });
                if (insert_it - out.begin() < static_cast<ptrdiff_t>(k)) {
                    if (out.size() == k) {
                        out.pop_back();
                    }
                    out.insert(insert_it, *first);
                }
            }
        }

        auto build(const std::vector<photon_type>::iterator first, const std::vector<photon_type>::iterator last, const size_t axis) -> void {
            if (last - first > 1) {
                std::sort(first, last, [axis](const auto p1, const auto p2) { return p1.position[axis] < p2.position[axis]; });
                const auto median = first + (last - first) / 2;
                build(first, median, (axis + 1) % 3);
                build(median + 1, last, (axis + 1) % 3);
            }
        }

        std::vector<photon_type> v;
};

template<typename T, typename Generator>
auto uvector3_distribution(Generator& prng) -> uvector3<T> {
    std::uniform_real_distribution<T> dist(-1.0f, 1.0f);
    return normalize(vector3<T>{dist(prng), dist(prng), dist(prng)});
}

auto ray_intersect(const uline3<float>& ray, const world_type& world) -> boost::optional<std::tuple<uvector3f, material_type, float>>
{
    const auto hit_sphere = [&]() -> boost::optional<std::tuple<uvector3f, material_type, float>> {
        const auto hit = ray_intersect(ray, world.sphere_objects.begin(), world.sphere_objects.end());
        if (std::get<0>(hit) != world.sphere_objects.end()) {
            const auto so = *std::get<0>(hit);
            const auto t = std::get<1>(hit);
            const auto n = normalize(ray(t) - so.mesh.c);
            return boost::optional<std::tuple<uvector3f, material_type, float>>(std::make_tuple(n, so.material, t));
        } else {
            return {};
        }
    }();

    const auto hit_parallelogram = [&]() -> boost::optional<std::tuple<uvector3f, material_type, float>> {
        const auto hit = ray_intersect(ray, world.parallelogram_objects.begin(), world.parallelogram_objects.end());
        if (std::get<0>(hit) != world.parallelogram_objects.end()) {
            const auto so = *std::get<0>(hit);
            const auto t = std::get<1>(hit);
            const auto n = [&]() {
                const auto n = cross(so.mesh.vx, so.mesh.vy);
                return normalize(dot(n, so.mesh.p - ray.p) < 0 ? n : -n);
            }();
            return boost::optional<std::tuple<uvector3f, material_type, float>>(std::make_tuple(n, so.material, t));
        } else {
            return {};
        }
    }();

    if (hit_sphere) {
        if (hit_parallelogram) {
            return std::get<2>(*hit_sphere) < std::get<2>(*hit_parallelogram) ? hit_sphere : hit_parallelogram;
        } else {
            return hit_sphere;
        }
    } else {
        return hit_parallelogram;
    }
}

auto photon_trace(const uline3<float>& ray, const rgb<float>& color, const world_type& world, std::mt19937& prng, const uint8_t max_depth) -> boost::optional<photon_type> {
    std::function<boost::optional<photon_type> (const uline3<float>&, const rgb<float>&, const uint8_t)> go =
        [&](const uline3<float>& ray, const rgb<float>& color, const uint8_t depth) mutable -> boost::optional<photon_type> {
            if (depth < max_depth) {
                const auto hit = ray_intersect(ray, world);
                if (hit) {
                    const auto normal = std::get<0>(*hit);
                    const auto material = std::get<1>(*hit);
                    const auto hitpos = ray(std::get<2>(*hit));

                    const float rr = std::uniform_real_distribution<float>(0.0f, 1.0f)(prng);
                    if (rr < material.diffuse_probability) {
                        const auto newray_position = hitpos + normal * 0.0001f;
                        while (true) {
                            const auto newray_direction = uvector3_distribution<float>(prng);
                            if (dot(newray_direction, normal) > 0.0f) {
                                const rgbf newray_color = color * material.brdf(-ray.v, newray_direction, normal);
                                return go({newray_position, newray_direction}, newray_color, depth + 1);
                            }
                        };
                    } else {
                        return boost::optional<photon_type>({hitpos, -ray.v, color, depth});
                    }
                } else {
                    return {};
                }
            } else {
                return {};
            }
        };

    return go(ray, color, 0);
}

auto generate_photon_map(const world_type& world, std::mt19937&& prng) -> kdtree {
    std::vector<photon_type> photons;

    for (const auto& point_light: world.point_lights) {
        for (size_t n = 0; n < point_light.num_photons; ++n) {
            const uline3f ray{point_light.position, uvector3_distribution<float>(prng)};

            if (const auto photon = photon_trace(ray, point_light.color / static_cast<float>(point_light.num_photons), world, prng, 5)) {
                photons.emplace_back(*photon);
            }
        }
    }

    BOOST_LOG_TRIVIAL(debug) << "Number of photons stored: " << photons.size();
    return kdtree(std::move(photons));
}

__global__ void render_cuda(const sphere_object_type* first_sphere_objects, const sphere_object_type* last_sphere_objects,
        const parallelogram_object_type* first_parallelogram_objects, const parallelogram_object_type* last_parallelogram_objects,
        const point_light_type* first_point_lights, const point_light_type* last_point_lights,
        const photon_type* first_photon, const photon_type* last_photon,
        const camera_type<float>& camera,
        rgba<float>* result, const size_t width, const size_t height);

auto render(const world_type& world, const camera_type<float>& camera, const size_t width, const size_t height) -> image2d<rgba<float>> {
    image2d<rgba<float>> result(width, height);

    const auto photons = generate_photon_map(world, std::mt19937(std::random_device()()));
#ifndef USE_CUDA
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t j = 0; j < height; ++j) {
        for (size_t i = 0; i < width; ++i) {
            const float x = static_cast<float>(i + 0.5) / static_cast<float>(width);
            const float y = static_cast<float>(j + 0.5) / static_cast<float>(height);
            const uline3f ray = { camera.position, normalize(camera.image_plane(x, y) - camera.position) };
            auto color = rgbf::black;

            const auto hit = ray_intersect(ray, world);
            if (hit) {
                const auto normal = std::get<0>(*hit);
                const auto material = std::get<1>(*hit);
                const auto hitpos = ray(std::get<2>(*hit));

                for (const auto& point_light: world.point_lights) {
                    const auto shadow_ray_direction = normalize(point_light.position - hitpos);
                    const auto offset = 0.001f;
                    const uline3f shadow_ray = { hitpos + shadow_ray_direction * offset, shadow_ray_direction };
                    const auto shadow_hit = ray_intersect(shadow_ray, world);
                    if (!shadow_hit || std::get<2>(*shadow_hit) > length(point_light.position - hitpos)) {
                        color += point_light.color * material.brdf(shadow_ray_direction, normalize(camera.position - hitpos), normal);
                    }
                }

                const auto nearest_photons = photons.knn(hitpos, 300);
                const auto r = distance(nearest_photons.back().position, hitpos);
                for (const auto photon: nearest_photons) {
                    color += photon.color * material.brdf(photon.direction, normalize(camera.position - hitpos), normal) * 400.0f / (float(M_PI) * r * r);
                }
            }
            result(i, j) = clamp(color * 256.0f, 0.0f, 255.0f);
        }
    }
#else
    render_cuda(world.sphere_objects.data(), world.sphere_objects.data() + world.sphere_objects.size(),
                world.parallelogram_objects.data(), world.parallelogram_objects.data() + world.parallelogram_objects.size(),
                world.point_lights.data(), world.point_lights.data() + world.point_lights.size(),
                photons.vector().data(), photons.vector().data() + photons.vector().size(),
                camera, &result(0, 0), width, height);
#endif

    return result;
}

auto main(const int, const char * const * const) -> int {
    const auto brdf_red = phong_type { .diffuse_color = {3.0f, 0.2f, 0.2f}, .specular_color = {1.0f, 1.0f, 1.0f}, .shininess = 8.0f };
    const auto brdf_green = phong_type { .diffuse_color = {0.0f, 1.0f, 0.2f}, .specular_color = {1.0f, 1.0f, 1.0f}, .shininess = 8.0f };
    const auto brdf_blue = phong_type { .diffuse_color = {0.4f, 0.4f, 1.0f}, .specular_color = {1.0f, 1.0f, 1.0f}, .shininess = 32.0f };
    const auto brdf_grey = phong_type { .diffuse_color = {0.5f, 0.5f, 0.5f}, .specular_color = {1.0f, 1.0f, 1.0f}, .shininess = 8.0f };
    const auto sphere_object = sphere_object_type { .mesh = {{0.0f, 0.0f, 0.0f}, 0.7f}, .material = { .diffuse_probability = 0.4f, .specular_probability = 0.2f, .brdf = brdf_blue } };
    const auto parallelogram_object1 = parallelogram_object_type { .mesh = {{1.8f, -2.0f, -2.0f}, {-0.1f, 4.0f, 0.0f}, {-0.1f, 0.0f, 4.0f}}, .material = { .diffuse_probability = 0.8f, .specular_probability = 0.2f, .brdf = brdf_red } };
    const auto parallelogram_object2 = parallelogram_object_type { .mesh = {{-4.0f, -0.9f, -4.0f}, {8.0f, 0.1f, 0.0f}, {0.0f, 0.1f, 8.0f}}, .material = { .diffuse_probability = 0.4f, .specular_probability = 0.4f, .brdf = brdf_grey } };
    const auto parallelogram_object3 = parallelogram_object_type { .mesh = {{-1.8f, -2.0f, -2.0f}, {0.1f, 4.0f, 0.0f}, {0.1f, 0.0f, 4.0f}}, .material = { .diffuse_probability = 0.8f, .specular_probability = 0.2f, .brdf = brdf_green } };
    const auto point_light = std::vector<point_light_type>{ {{0.5f, 2.5f, -1.0f}, {0.8f, 0.8f, 0.8f}, 12800} };

    const auto world = world_type { .sphere_objects = {sphere_object}, .parallelogram_objects = {parallelogram_object1, parallelogram_object2, parallelogram_object3}, .point_lights = {point_light} };
    const auto camera = camera_type<float> { .position = {0.0f, 0.0f, -3.0f}, .image_plane = { .p = {-1.5f, -1.5f, 0.0f}, .vx = {3.0f, 0.0f, 0.0f}, .vy = {0.0f, 3.0f, 0.0f} } };

    const image2d<rgba<float>> image = render(world, camera, 1024, 1024);
    std::ofstream output_file("output.tga");
    save_tga(image, std::ostreambuf_iterator<char>(output_file));

    return 0;
}
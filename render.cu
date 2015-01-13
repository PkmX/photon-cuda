#include <cstdint>
#include <iostream>
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <limits>
#include <cuda_runtime.h>

#include "camera.hpp"
#include "line3.hpp"
#include "parallelogram.hpp"
#include "point3.hpp"
#include "rgba.hpp"
#include "rgb.hpp"
#include "sphere.hpp"
#include "uvector3.hpp"

#define cudaCheckError() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
} while(0)

struct point_light_type {
    point3f position;
    rgb<float> color;
    size_t num_photons;
};

struct phong_type {
    rgb<float> diffuse_color, specular_color;
    float shininess;

    __host__ __device__ auto operator()(const uvector3f& wi, const uvector3f& wo, const uvector3f& n) const -> rgb<float> {
        const auto nl = dot(n, wi);
        const auto r = normalize(2 * dot(wi, n) * n - wi);
        const auto re = dot(r, wo);

        return (nl > 0.0f ? diffuse_color * nl : rgbf {0.0f, 0.0f, 0.0f}) + (re > 0.0f ? specular_color * std::pow(re, shininess) : rgbf {0.0f, 0.0f, 0.0f});
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

struct photon_type {
    point3f position;
    uvector3f direction = uvector3f({1.0f, 0.0f, 0.0f});
    rgb<float> color;
    std::uint8_t depth;
};

struct intersect_result_type {
    uvector3f normal;
    material_type material;
    float t;
};

__device__ intersect_result_type ray_intersect(const uline3f& r,
                                               const sphere_object_type* first_sphere_objects, const sphere_object_type* last_sphere_objects,
                                               const parallelogram_object_type* first_parallelogram_objects, const parallelogram_object_type* last_parallelogram_objects)
{
    auto hit_normal = uvector3f({ 1.0f, 0.0f, 0.0f });
    auto hit_material = material_type {};
    auto hit_t = 1.0f / 0.0f;

    for (auto it = first_sphere_objects; it != last_sphere_objects; ++it) {
        const auto& s = it->mesh;
        const auto c = s.c - r.p;
        const auto b = dot(r.v, c);
        const auto d = s.r * s.r - (length2(c) - b * b);

        if (d >= 0) {
            const auto t0 = b - sqrt(d), t1 = b + sqrt(d);
            if (t0 > 0 && t0 <= t1) {
                if (t0 < hit_t) {
                    hit_normal = normalize(r(t0) - s.c);
                    hit_material = it->material;
                    hit_t = t0;
                }
            } else if (t1 > 0) {
                if (t1 < hit_t) {
                    hit_normal = normalize(r(t1) - s.c);
                    hit_material = it->material;
                    hit_t = t1;
                }
            }
        }
    }

    for (auto it = first_parallelogram_objects; it != last_parallelogram_objects; ++it) {
        const auto& p = it->mesh;

        const auto n = [&]() {
            const auto n = cross(p.vx, p.vy);
            return normalize(dot(n, p.p - r.p) < 0 ? n : -n);
        }();
        const auto vn = dot(r.v, n);

        if (vn < 0) {
            const auto t = dot(p.p - r.p, static_cast<vector3f>(n)) / dot(r.v, static_cast<vector3f>(n));
            const auto rt = r(t);
            const auto sx = scalar_proj(rt - p.p, p.vx);
            const auto sy = scalar_proj(rt - p.p, p.vy);

            if (0 <= sx && sx <= length(p.vx) && 0 <= sy && sy <= length(p.vy)) {
                if (t < hit_t) {
                    hit_normal = [&]() {
                        const auto n = cross(p.vx, p.vy);
                        return normalize(dot(n, p.p - r.p) < 0 ? n : -n);
                    }();
                    hit_material = it->material;
                    hit_t = t;
                }
            }
        }
    }

    return {hit_normal, hit_material, hit_t};
}

__device__ void knn_cuda_impl(const point3f& p, const size_t k, const photon_type* first, const photon_type* last, const size_t axis, photon_type* results, size_t& num_results) {
    if (last - first > 1) {
        const auto median = first + (last - first) / 2;
        const auto nearer_first = p[axis] < median->position[axis] ? first : median;
        const auto nearer_last = p[axis] < median->position[axis] ? median : last;
        const auto farther_first = p[axis] < median->position[axis] ? median : first;
        const auto farther_last = p[axis] < median->position[axis] ? last : median;
        const auto next_axis = (axis + 1) % 3;

        knn_cuda_impl(p, k, nearer_first, nearer_last, next_axis, results, num_results);
        if (num_results < k || (num_results > 0 ? distance(results[num_results - 1].position, p) > abs(median->position[axis] - p[axis]) : true)) {
            knn_cuda_impl(p, k, farther_first, farther_last, next_axis, results, num_results);
        }
    } else if (last - first == 1) {
        const auto dist = distance(p, first->position);
        const auto ipos = [&]() {
            auto pos = results + num_results;
            while (pos != results && distance(p, (pos - 1)->position) > dist) --pos;
            return pos;
        }();

        if (ipos < results + k) {
            for (auto it = results + (k - 1 < num_results ? k - 1 : num_results); it > ipos; it -= 1) {
                *it = *(it - 1);
            }
            if (num_results < k) num_results += 1;
            *ipos = *first;
        }
    }
}

__global__ void render_cuda_kernel(const sphere_object_type* first_sphere_objects, const sphere_object_type* last_sphere_objects,
                                   const parallelogram_object_type* first_parallelogram_objects, const parallelogram_object_type* last_parallelogram_objects,
                                   const point_light_type* first_point_lights, const point_light_type* last_point_lights,
                                   const photon_type* first_photon, const photon_type* last_photon,
                                   const camera_type<float> camera,
                                   rgba<float>* result, const size_t width, const size_t height) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t j = threadIdx.y + blockIdx.y * blockDim.y;

    const auto x = static_cast<float>(i + 0.5) / static_cast<float>(width);
    const auto y = static_cast<float>(j + 0.5) / static_cast<float>(height);
    const auto ray = uline3f { camera.position, normalize(camera.image_plane(x, y) - camera.position) };
    auto color = rgbf {0.0f, 0.0f, 0.0f};

    const auto hit = ray_intersect(ray, first_sphere_objects, last_sphere_objects, first_parallelogram_objects, last_parallelogram_objects);
    if (hit.t < 1.0f / 0.0f) {
        const auto normal = hit.normal;
        const auto material = hit.material;
        const auto hitpos = ray(hit.t);

        for (auto it = first_point_lights; it != last_point_lights; ++it) {
            const auto& point_light = *it;
            const auto shadow_ray_direction = normalize(point_light.position - hitpos);
            const auto offset = 0.001f;
            const auto shadow_ray = uline3f { hitpos + shadow_ray_direction * offset, shadow_ray_direction };
            const auto shadow_hit = ray_intersect(shadow_ray, first_sphere_objects, last_sphere_objects, first_parallelogram_objects, last_parallelogram_objects);
            if (shadow_hit.t == 1.0f / 0.0f || shadow_hit.t > length(point_light.position - hitpos)) {
                color += point_light.color * material.brdf(shadow_ray_direction, normalize(camera.position - hitpos), normal);
            }
        }

        photon_type nearest_photons[300];
        size_t num_nearest_photons = 0;
        knn_cuda_impl(hitpos, 300, first_photon, last_photon, 0, nearest_photons, num_nearest_photons);

        if (num_nearest_photons > 0) {
            const auto r = distance(nearest_photons[num_nearest_photons - 1].position, hitpos);
            for (auto photon = nearest_photons; photon < nearest_photons + num_nearest_photons; ++photon) {
                color += photon->color * material.brdf(photon->direction, normalize(camera.position - hitpos), normal) * 800.0f / (float(M_PI) * r * r);
            }
        }
    }

    result[i + j * width] = clamp(color * 256.0f, 0.0f, 255.0f);
}

void render_cuda(const sphere_object_type* first_sphere_objects, const sphere_object_type* last_sphere_objects,
                 const parallelogram_object_type* first_parallelogram_objects, const parallelogram_object_type* last_parallelogram_objects,
                 const point_light_type* first_point_lights, const point_light_type* last_point_lights,
                 const photon_type* first_photon, const photon_type* last_photon,
                 const camera_type<float>& camera,
                 rgbaf* result, const size_t width, const size_t height)
{
    const size_t num_sphere_objects = last_sphere_objects - first_sphere_objects;
    const size_t num_parallelogram_objects = last_parallelogram_objects - first_parallelogram_objects;
    const size_t num_point_lights = last_point_lights - first_point_lights;
    const size_t num_photons = last_photon - first_photon;

    cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024);
    cudaCheckError();

    sphere_object_type* device_sphere_objects;
    cudaMalloc(&device_sphere_objects, num_sphere_objects * sizeof(sphere_object_type));
    cudaCheckError();
    cudaMemcpy(device_sphere_objects, first_sphere_objects, num_sphere_objects * sizeof(sphere_object_type), cudaMemcpyHostToDevice);
    cudaCheckError();

    parallelogram_object_type* device_parallelogram_objects;
    cudaMalloc(&device_parallelogram_objects, num_parallelogram_objects * sizeof(parallelogram_object_type));
    cudaCheckError();
    cudaMemcpy(device_parallelogram_objects, first_parallelogram_objects, num_parallelogram_objects * sizeof(parallelogram_object_type), cudaMemcpyHostToDevice);
    cudaCheckError();

    point_light_type* device_point_lights;
    cudaMalloc(&device_point_lights, num_point_lights * sizeof(point_light_type));
    cudaCheckError();
    cudaMemcpy(device_point_lights, first_point_lights, num_point_lights * sizeof(point_light_type), cudaMemcpyHostToDevice);
    cudaCheckError();

    photon_type* device_photons;
    cudaMalloc(&device_photons, num_photons * sizeof(photon_type));
    cudaCheckError();
    cudaMemcpy(device_photons, first_photon, num_photons * sizeof(photon_type), cudaMemcpyHostToDevice);
    cudaCheckError();

    rgbaf* device_result;
    cudaMalloc(&device_result, width * height * sizeof(rgbaf));
    cudaCheckError();
    render_cuda_kernel<<<dim3(width / 8, height / 8), dim3(8, 8)>>>(device_sphere_objects, device_sphere_objects + num_sphere_objects,
                                                                    device_parallelogram_objects, device_parallelogram_objects + num_parallelogram_objects,
                                                                    device_point_lights, device_point_lights + num_point_lights,
                                                                    device_photons, device_photons + num_photons,
                                                                    camera, device_result, width, height);
    cudaCheckError();

    cudaMemcpy(result, device_result, width * height * sizeof(rgbaf), cudaMemcpyDeviceToHost);
    cudaCheckError();
}
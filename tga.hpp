#ifndef TGA_H
#define TGA_H

#include <cstdint>
#include <iostream>
#include <algorithm>
#include "image2d.hpp"

#pragma pack(1)
struct tga_header {
    uint8_t     id_length;
    uint8_t     colormap_type;
    uint8_t     image_type;
    uint16_t    colormap_offset;
    uint16_t    colormap_size;
    uint8_t     colormap_bpp;
    uint16_t    xorigin;
    uint16_t    yorigin;
    uint16_t    width;
    uint16_t    height;
    uint8_t     bpp;
    uint8_t     descriptor;
};
#pragma pack()

template<typename T, typename OutputIterator>
void save_tga(const image2d<T>& img, OutputIterator out) {
    const tga_header header = {
        .id_length = 0,
        .colormap_type = 0,
        .image_type = 2,
        .colormap_offset = 0,
        .colormap_size = 0,
        .colormap_bpp = 0,
        .xorigin = 0,
        .yorigin = 0,
        .width = static_cast<uint16_t>(img.width()),
        .height = static_cast<uint16_t>(img.height()),
        .bpp = 24,
        .descriptor = 0
    };

    std::copy(reinterpret_cast<const char*>(&header), reinterpret_cast<const char*>(&header) + sizeof(header), out);
    
    // For each color in the image.
    for (const auto c: img) {
        *out++ = c[2];
        *out++ = c[1];
        *out++ = c[0];
    }
}

#endif // TGA_H

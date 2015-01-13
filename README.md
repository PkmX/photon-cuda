# Photon Mapping on CUDA

Project for Parallel Programming (2014 Fall)

## Dependencies

* cmake >= 2.8
* boost.log >= 1.57.0
* cuda >= 6.5
* C++14 compiler (gcc >= 4.9 / clang >= 3.5)

## Building

    cmake -DCMAKE_BUILD_TYPE=Release .
    make
    
## Running

    ./photon-cuda
    ${IMAGE_VIEWER} output.tga

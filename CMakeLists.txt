cmake_minimum_required (VERSION 2.8)
project (photon-cuda)

find_package(Boost 1.57.0 COMPONENTS log REQUIRED)
find_package(CUDA 6.5 REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_ALL_DYN_LINK -std=c++14 -Wall -Wextra -pedantic -Wno-c99-extensions -pthread")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -pthread -Xcompiler -isystem -Xcompiler ${CUDA_INCLUDE_DIRS})
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_ALL_DYN_LINK -std=c++14 -Wall -Wextra -pthread -fopenmp")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -pthread -Xcompiler -fopenmp -Xcompiler -isystem -Xcompiler ${CUDA_INCLUDE_DIRS})
else()
    message( FATAL_ERROR "Can only be built with gcc or clang." )
endif()

cuda_compile(RENDER_O render.cu)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
cuda_add_executable(photon-cuda main.cpp ${RENDER_O})
target_link_libraries(photon-cuda ${Boost_LIBRARIES})
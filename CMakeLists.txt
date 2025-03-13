cmake_minimum_required(VERSION 3.28)

project(pchromatic VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
# link_libraries(CUDA::toolkit CUDA::cudart)

include_directories(include)

add_subdirectory(tests)

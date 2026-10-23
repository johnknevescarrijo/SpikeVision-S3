cmake_minimum_required(VERSION 3.16)

# Suporte híbrido transparente: ESP-IDF Toolchain ou Host Linux CMake
if(DEFINED ENV{IDF_PATH})
    include($ENV{IDF_PATH}/tools/cmake/project.cmake)
    project(spikevision_s3)
else()
    project(spikevision_s3_host C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    add_compile_options(-O3 -Wall -Wextra)
    include_directories(include test_samples main)
    add_executable(spikevision_s3_host main/main.cpp main/snn_engine.cpp)
endif()

cmake_minimum_required(VERSION 3.21)

project(supercomputers)

set(COMMON_PATH ${CMAKE_SOURCE_DIR}/src/common)

add_subdirectory(src/openmp_single)

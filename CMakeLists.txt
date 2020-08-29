cmake_minimum_required(VERSION 3.1)

project(TSDFFuion)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem)

include_directories(./include/)

add_executable(tsdf-fusion-cpu src/tsdf_fusion_cpu.cpp)
target_link_libraries(tsdf-fusion-cpu ${OpenCV_LIBS})

add_executable(tsdf-fusion-dual-cam src/tsdf_fusion_dual_cam.cpp)
target_link_libraries(tsdf-fusion-dual-cam ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
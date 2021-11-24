cmake_minimum_required(VERSION 3.1)

project(TSDFFuion)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem)
find_package(PCL REQUIRED REQUIRED COMPONENTS io registration segmentation filters search)
find_package(Eigen3 REQUIRED)

include_directories(
    ./include/
    ${PCL_INCLUDE_LIBS}
)


add_executable(tsdf-fusion-cpu src/tsdf_fusion_cpu.cpp)
target_link_libraries(tsdf-fusion-cpu ${OpenCV_LIBS})

add_executable(tsdf-fusion-plane-segmentation src/tsdf_fusion_plane_segmentation.cpp)
target_link_libraries(tsdf-fusion-plane-segmentation ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES})

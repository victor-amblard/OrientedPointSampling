cmake_minimum_required(VERSION 3.5)

project(orientedPointSampling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization segmentation filters )
find_package(Boost REQUIRED COMPONENTS filesystem serialization regex timer date_time thread)
 file(GLOB source_files
            "include/*.h"
            "src/*.cpp"
    )
include_directories("include")
add_definitions(${PCL_DEFINITIONS})
add_executable(orientedPointSampling ${source_files})
target_link_libraries (orientedPointSampling ${PCL_LIBRARIES})                                                                                                                                  
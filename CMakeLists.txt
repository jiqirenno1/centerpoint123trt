cmake_minimum_required(VERSION 3.10)
project(centerpointTrt)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/common)

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorRT
include_directories(/home/ubuntu/mysoftware/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/include)
link_directories(/home/ubuntu/mysoftware/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/lib)

#include_directories(/home/ubuntu/mysoftware/TensorRT-8.0.1.6/include)
#link_directories(/home/ubuntu/mysoftware/TensorRT-8.0.1.6/lib)
#include_directories(${PROJECT_SOURCE_DIR}/ScatterNDPlugin.h)
#cuda_add_library(ScatterNDPlugin SHARED ${PROJECT_SOURCE_DIR}/ScatterNDPlugin.cu)
#target_link_libraries(ScatterNDPlugin nvinfer cudart)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)

file(GLOB coms common/*.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE src src/*.cpp)
message("print:"${src})

cuda_add_library(centerpoint  SHARED ${src} src/ScatterNDPlugin.cu ${coms})
target_link_libraries(centerpoint nvinfer cudart nvonnxparser -lpthread ${PCL_LIBRARIES} ${OpenCV_LIBS})

cuda_add_executable(cptest test_fusion.cpp)
target_link_libraries(cptest centerpoint)

cuda_add_executable(vis test_vis.cpp)
target_link_libraries(vis centerpoint)

cuda_add_executable(track test_tracking.cpp)
target_link_libraries(track centerpoint)
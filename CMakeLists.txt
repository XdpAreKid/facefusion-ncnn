cmake_minimum_required(VERSION 3.0.0)
project(facefusion-ncnn VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/src
)
add_library(facefusion src/utils.cpp src/yoloface.cpp src/faceRecognize.cpp src/faceSwap.cpp src/codeFormer/codeformer.cpp src/codeFormer/encoder.cpp src/codeFormer/generator.cpp)
add_executable(facefusion-ncnn main.cpp)
target_link_libraries(facefusion-ncnn ncnn ${OpenCV_LIBS} facefusion)
add_subdirectory(test)
# add_executable(facefusion-ncnn main.cpp src/utils.cpp src/yoloface.cpp)
# target_link_libraries(facefusion-ncnn ncnn ${OpenCV_LIBS})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

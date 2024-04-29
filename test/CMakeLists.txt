project(test)

add_executable(test_yolo test_yolo.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/)
target_link_libraries(test_yolo
    facefusion
    ${OpenCV_LIBS}
    ncnn
)
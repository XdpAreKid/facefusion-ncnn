

#include "yoloface.h"
#include <cstdio>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <image_path>\n", argv[0]);
    return 1;
  }

  // Load the image
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (image.empty()) {
    printf("Could not read the image: %s\n", argv[1]);
    return 1;
  }

  YoloFace face;
  face.load("./", true);
  std::vector<Bbox> objects;
  face.detect(image, objects);
  face.draw(image, objects);
  cv::imwrite("detect_result.png", image);
  return 0;
}
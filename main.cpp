#include "codeFormer/codeformer.h"
#include "faceRecognize.h"
#include "faceSwap.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "yoloface.h"
#include <cstdio>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "Hello, from facefusion-ncnn!\n";
  if (argc != 3) {
    printf("Usage:%s <source_image> <style_image> \n", argv[0]);
    return -1;
  }
  YoloFace faceModel;
  faceModel.load("../model", true);

  faceRec recModel;
  recModel.load("../model", true);

  faceSwap swapModel;
  swapModel.load("../model", false);

  // CodeFormer codeFormerModel;
  // codeFormerModel.Load("../model");

  std::vector<Bbox> source_objects;
  std::vector<Bbox> target_objects;

  //   face.detect(image, objects);
  cv::Mat source_img = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat style_img = cv::imread(argv[2], cv::IMREAD_COLOR);
  faceModel.detect(source_img, source_objects);
  faceModel.detect(style_img, target_objects);

  if (target_objects.size() >= 2) {
    printf("target face > 2, but only supported one\n");
  } else if (target_objects.empty()) {
    printf("target face is no face detect\n");
    return -1;
  }
  std::vector<float> style_embedding;
  recModel.detect(style_img, target_objects.front(), style_embedding);
  cv::Mat dst;
  // TODO: call facefusion-ncnn
  swapModel.swap(source_img, style_embedding,
                 transformLandMark(source_objects.front().pts), dst);

  cv::imwrite("result.jpg", dst);

  return 0;
}

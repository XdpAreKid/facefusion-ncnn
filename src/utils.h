#ifndef UTILS
#define UTILS
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
typedef struct {
  float x;
  float y;
  float score;
} Landmarks;

typedef struct {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float prob;
  std::vector<Landmarks> pts;
} Bbox;

float GetIoU(const Bbox box1, const Bbox box2);
std::vector<int> nms(std::vector<Bbox> boxes, std::vector<float> confidences,
                     const float nms_thresh);

std::vector<cv::Point2f> transformLandMark(const std::vector<Landmarks> &lands);

cv::Mat
warp_face_by_face_landmark_5(const cv::Mat temp_vision_frame, cv::Mat &crop_img,
                             const std::vector<cv::Point2f> face_landmark_5,
                             const std::vector<cv::Point2f> normed_template,
                             const cv::Size crop_size);
cv::Mat create_static_box_mask(const int *crop_size, const float face_mask_blur,
                               const int *face_mask_padding);
cv::Mat paste_back(cv::Mat temp_vision_frame, cv::Mat crop_vision_frame,
                   cv::Mat crop_mask, cv::Mat affine_matrix);
cv::Mat blend_frame(cv::Mat temp_vision_frame, cv::Mat paste_vision_frame,
                    const int FACE_ENHANCER_BLEND = 80);
#endif
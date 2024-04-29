#ifndef _YOLOV8FACE_H_
#define _YOLOV8FACE_H_
#include "utils.h"
#include <ncnn/net.h>
#include <opencv2/core/types.hpp>

class YoloFace {
public:
  YoloFace();

  int load(std::string model_path, bool use_gpu = false);

  int detect(const cv::Mat &rgb, std::vector<Bbox> &objects,
             float prob_threshold = 0.25f, float nms_threshold = 0.45f);

  int draw(cv::Mat &rgb, const std::vector<Bbox> &objects);

private:
  ncnn::Net yoloface;

  int target_size;
  float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  bool lite_t = false;
};

#endif // _YOLOV8FACE_H_
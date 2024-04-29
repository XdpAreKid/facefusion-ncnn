#ifndef _FACESWAP_H_
#define _FACESWAP_H_

#include "opencv2/core/mat.hpp"
#include <ncnn/net.h>
#include <string>
#include <vector>
class faceSwap {
public:
  faceSwap();
  ~faceSwap();
  void load(std::string model_path, bool use_gpu = false);
  void swap(const cv::Mat &target, const std::vector<float> &style_embedding,
            const std::vector<cv::Point2f> &target_landmark_5, cv::Mat &dst);

private:
  float *model_matrix;
  const int len_feature = 512;

  const float FACE_MASK_BLUR = 0.3;
  const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};
  const float normVal[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  ncnn::Net mNet;
};

#endif // _FACESWAP_H_
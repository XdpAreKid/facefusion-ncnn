#ifndef _FACERECOGNIZE_H_
#define _FACERECOGNIZE_H_

#include "opencv2/core/mat.hpp"
#include "utils.h"
#include <ncnn/net.h>
class faceRec {
public:
  faceRec();
  void load(std::string model_path, bool use_gpu = false);
  void detect(const cv::Mat &img, const Bbox &obj, std::vector<float> &result);

private:
  void postprocess(ncnn::Mat &out, std::vector<float> &result);

private:
  ncnn::Net mNet;

  const int mInputSize = 112;
  float mMeanVal[3] = {127.5, 127.5, 127.5};
  float mNormVal[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
};

#endif // _FACERECOGNIZE_H_

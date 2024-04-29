#include "faceRecognize.h"
#include "opencv2/core/types.hpp"
#include "utils.h"
#include <vector>

faceRec::faceRec() {}

void faceRec::load(std::string model_path, bool use_gpu) {
  mNet.clear();

  mNet.opt = ncnn::Option();
#if NCNN_VULKAN
  mNet.opt.use_vulkan_compute = use_gpu;
#endif

  mNet.load_param((model_path + "/arcface.param").c_str());
  mNet.load_model((model_path + "/arcface.bin").c_str());
  return;
}

const static std::vector<cv::Point2f> faceTem = {
    cv::Point2f(38.29459984, 51.69630032),
    cv::Point2f(73.53180016, 51.50140016), cv::Point2f(56.0252, 71.73660032),
    cv::Point2f(41.54929968, 92.36549952),
    cv::Point2f(70.72989952, 92.20409968)};

void faceRec::detect(const cv::Mat &img, const Bbox &obj,
                     std::vector<float> &result) {

  cv::Mat cropImg;
  warp_face_by_face_landmark_5(img, cropImg, transformLandMark(obj.pts),
                               faceTem, cv::Size(mInputSize, mInputSize));
  ncnn::Mat in = ncnn::Mat::from_pixels(cropImg.data, ncnn::Mat::PIXEL_BGR,
                                        cropImg.cols, cropImg.rows);
  in.substract_mean_normalize(mMeanVal, mNormVal);

  ncnn::Extractor ex = mNet.create_extractor();
  ex.input("input.1", in);
  ncnn::Mat out;
  ex.extract("683", out);

  result.resize(out.cstep);
  memcpy(result.data(), out.data, out.cstep * sizeof(float));

  return;
}

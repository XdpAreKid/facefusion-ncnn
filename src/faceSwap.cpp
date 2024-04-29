#include "faceSwap.h"
#include "opencv2/imgcodecs.hpp"
#include "utils.h"
#include <ncnn/mat.h>
#include <vector>

const static std::vector<cv::Point2f> faceTem2 = {
    cv::Point2f(46.29459968, 51.69630032),
    cv::Point2f(81.53180032, 51.50140016),
    cv::Point2f(64.02519936, 71.73660032),
    cv::Point2f(49.54930048, 92.36549952),
    cv::Point2f(78.72989952, 92.20409968)};

void faceSwap::load(std::string model_path, bool use_gpu) {
  mNet.clear();
  mNet.opt = ncnn::Option();
#if NCNN_VULKAN
  mNet.opt.use_vulkan_compute = use_gpu;
#endif

  mNet.load_param((model_path + "/inswapper_opt.param").c_str());
  mNet.load_model((model_path + "/inswapper_opt.bin").c_str());
}

void faceSwap::swap(const cv::Mat &target,
                    const std::vector<float> &style_embedding,
                    const std::vector<cv::Point2f> &target_landmark_5,
                    cv::Mat &dst) {

  std::vector<float> input_embedding;
  float linalg_norm = 0;
  for (int i = 0; i < this->len_feature; i++) {
    linalg_norm += powf(style_embedding[i], 2);
  }
  linalg_norm = sqrt(linalg_norm);
  input_embedding.resize(this->len_feature);
  for (int i = 0; i < this->len_feature; i++) {
    float sum = 0;
    for (int j = 0; j < this->len_feature; j++) {
      sum +=
          (style_embedding[j] * this->model_matrix[j * this->len_feature + i]);
    }
    input_embedding[i] = sum / linalg_norm;
  }

  cv::Mat cropImg;
  cv::Mat affine_matrix, box_mask;

  affine_matrix = warp_face_by_face_landmark_5(
      target, cropImg, target_landmark_5, faceTem2, cv::Size(128, 128));
  const int crop_size[2] = {cropImg.cols, cropImg.rows};

  box_mask = create_static_box_mask(crop_size, this->FACE_MASK_BLUR,
                                    this->FACE_MASK_PADDING);
  ncnn::Mat input = ncnn::Mat::from_pixels(cropImg.data, ncnn::Mat::PIXEL_BGR,
                                           cropImg.cols, cropImg.rows);

  input.substract_mean_normalize(0, normVal);
  ncnn::Mat source(512, input_embedding.data());
  ncnn::Extractor ex = mNet.create_extractor();
  ex.input("target", input);
  ex.input("source", source);
  ncnn::Mat out;
  ex.extract("output", out);

  int channel_step = out.h * out.w;

  cv::Mat bmat(out.h, out.w, CV_32FC1, out.data);
  cv::Mat gmat(out.h, out.w, CV_32FC1, (float *)out.data + channel_step);
  cv::Mat rmat(out.h, out.w, CV_32FC1, (float *)out.data + 2 * channel_step);

  rmat *= 255.f;
  gmat *= 255.f;
  bmat *= 255.f;
  rmat.setTo(0, rmat < 0);
  rmat.setTo(255, rmat > 255);
  gmat.setTo(0, gmat < 0);
  gmat.setTo(255, gmat > 255);
  bmat.setTo(0, bmat < 0);
  bmat.setTo(255, bmat > 255);

  std::vector<cv::Mat> channel_mats(3);

  channel_mats[0] = bmat;
  channel_mats[1] = gmat;
  channel_mats[2] = rmat;
  cv::Mat result;
  merge(channel_mats, result);

  box_mask.setTo(0, box_mask < 0);
  box_mask.setTo(1, box_mask > 1);
  dst = paste_back(target, result, box_mask, affine_matrix);
}

faceSwap::faceSwap() {
  const int length = this->len_feature * this->len_feature;
  this->model_matrix = new float[length];
  FILE *fp = fopen("model_matrix.bin", "rb");
  fread(this->model_matrix, sizeof(float), length, fp); // 导入数据
  fclose(fp);                                           // 关闭文件
}

faceSwap::~faceSwap() {
  delete[] this->model_matrix;
  this->model_matrix = nullptr;
}
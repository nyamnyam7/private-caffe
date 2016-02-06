#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "caffe/layers/image_preprocessing_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void ImagePreprocessingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 1);

  method_ = this->layer_param_.imgproc_param().method();
  srand((unsigned)time(0));
}

template <typename Dtype>
void ImagePreprocessingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom[0]->num_axes(), 4);

  int n = bottom[0]->shape(0);
  int c = bottom[0]->shape(1);
  int h = bottom[0]->shape(2);
  int w = bottom[0]->shape(3);

  vector<int> shape;
  shape.push_back(n);
  shape.push_back(c);
  shape.push_back(h);
  shape.push_back(w);

  top[0]->Reshape(shape);
}

//////////////////// affine

static void affine(cv::Mat& input_img, cv::Mat& output_img)
{
  using namespace cv;

  Mat warp_dst;
  warp_dst = Mat::zeros(input_img.rows, input_img.cols, input_img.type());
  Mat rot_mat(2, 3, CV_32FC1);
  Mat warp_mat(2, 3, CV_32FC1);

  //Random translation
  Point2f srcTri[3];
  Point2f dstTri[3];

  srcTri[0] = Point2f(0, 0);
  srcTri[1] = Point2f(input_img.cols - 1, 0);
  srcTri[2] = Point2f(0, input_img.rows - 1);

  float x = (-10 + (21 * rand() / (RAND_MAX + 1.0))) / 100.0;
  float y = (-10 + (21 * rand() / (RAND_MAX + 1.0))) / 100.0;

  dstTri[0] = Point2f(input_img.cols*x, input_img.rows*y);
  dstTri[1] = Point2f(input_img.cols*(1.0 + x), input_img.rows*y);
  dstTri[2] = Point2f(input_img.cols*x, input_img.rows*(1.0 + y));

  warp_mat = getAffineTransform(srcTri, dstTri);
  warpAffine(input_img, warp_dst, warp_mat, warp_dst.size());

  //Random rotation and scaling
  Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
  double angle = -5 + (11 * rand() / (RAND_MAX + 1.0));
  double scale = (9 + (3 * rand() / (RAND_MAX + 1.0))) / 10.0;

  rot_mat = getRotationMatrix2D(center, angle, scale);
  warpAffine(warp_dst, output_img, rot_mat, warp_dst.size());
}


template <typename Dtype>
void ImagePreprocessingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int n = bottom[0]->shape(0);
  int c = bottom[0]->shape(1);
  int h = bottom[0]->shape(2);
  int w = bottom[0]->shape(3);

  vector<cv::Mat> inputs;
  vector<cv::Mat> outputs;

  for (int idx=0; idx < n; idx++) {
    vector<cv::Mat> channels(3);
    cv::Mat img;
    channels[0] = cv::Mat(h, w, CV_32FC1, bottom[0]->mutable_cpu_data() + (idx * c + 0) * h * w);
    channels[1] = cv::Mat(h, w, CV_32FC1, bottom[0]->mutable_cpu_data() + (idx * c + 1) * h * w);
    channels[2] = cv::Mat(h, w, CV_32FC1, bottom[0]->mutable_cpu_data() + (idx * c + 2) * h * w);
    cv::merge(channels, img);
    inputs.push_back(img);
  }

  // preprocessing
  for (int idx = 0; idx < n; idx++) {
    if (method_ == "affine") {
      cv::Mat output;
      affine(inputs[idx], output);
      outputs.push_back(output);
    } else if (method_ == "validation") {
      outputs.push_back(inputs[idx]);
    }
      else {
      cout << "Unknown preprocessing method: " << method_ << endl;
      outputs.push_back(inputs[idx]);
    }
  }

  for (int idx=0; idx < n; idx++) {
    vector<cv::Mat> channels(3);
    cv::split(outputs[idx], channels);
    for (int j=0; j<3; j++)
        memcpy(top[0]->mutable_cpu_data() + (idx * c + j) * h * w, channels[j].data, sizeof(float) * h * w);
  }

}

template <typename Dtype>
void ImagePreprocessingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

#ifdef CPU_ONLY
STUB_GPU(ImagePreprocessingLayer);
#endif

INSTANTIATE_CLASS(ImagePreprocessingLayer);
REGISTER_LAYER_CLASS(ImagePreprocessing);

}  // namespace caffe

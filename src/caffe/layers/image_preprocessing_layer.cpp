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
  Point2f srcQuad[4];
  Point2f dstQuad[4];

  float x = (-10 + (21 * rand() / (RAND_MAX + 1.0))) / 100.0 * 48;
  float y = (-10 + (21 * rand() / (RAND_MAX + 1.0))) / 100.0 * 48;

  if (x >= 0){
	if (y >= 0)
	{
		srcQuad[0] = Point2f(x, y);
		srcQuad[1] = Point2f(input_img.cols-1, y);
		srcQuad[2] = Point2f(x, input_img.rows - 1);
		srcQuad[3] = Point2f(input_img.cols - 1, input_img.rows - 1);
	}
	else
	{
		srcQuad[0] = Point2f(x, 0);
		srcQuad[1] = Point2f(input_img.cols - 1, 0);
		srcQuad[2] = Point2f(x, input_img.rows + y);
		srcQuad[3] = Point2f(input_img.cols - 1, input_img.rows + y);
	}
  }
  else
  {
	if (y >= 0)
	{
		srcQuad[0] = Point2f(0, y);
		srcQuad[1] = Point2f(input_img.cols + x, y);
		srcQuad[2] = Point2f(0, input_img.rows - 1);
		srcQuad[3] = Point2f(input_img.cols + x, input_img.rows - 1);
	}
	else
	{
		srcQuad[0] = Point2f(0, 0);
		srcQuad[1] = Point2f(input_img.cols + x, 0);
		srcQuad[2] = Point2f(0, input_img.rows + y);
		srcQuad[3] = Point2f(input_img.cols + x, input_img.rows + y);
	}
  }

  dstQuad[0] = Point2f(0, 0);
  dstQuad[1] = Point2f(input_img.cols - 1, 0);
  dstQuad[2] = Point2f(0, input_img.rows - 1);
  dstQuad[3] = Point2f(input_img.cols - 1, input_img.rows - 1);

  Mat trans_mat(2, 4, CV_32FC1);
  trans_mat = getPerspectiveTransform(srcQuad, dstQuad);
  warpPerspective(input_img, output_img, trans_mat, Size(48, 48));
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

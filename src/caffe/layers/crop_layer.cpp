#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CropParameter crop_param = this->layer_param_.crop_param();

  CHECK(crop_param.has_num_classes()) << "num_classes should be defined";
  num_classes_ = crop_param.num_classes();

  x1a_ = crop_param.x1a();
  x1b_ = crop_param.x1b();
  x2a_ = crop_param.x2a();
  x2b_ = crop_param.x2b();
  y1a_ = crop_param.x1a();
  y1b_ = crop_param.x1b();
  y2a_ = crop_param.x2a();
  y2b_ = crop_param.x2b();

  resample_ = crop_param.resample();
  random_ = crop_param.random();
  
  if (random_) {
    CHECK(x1a_ <= x1b_) << "Size constraint";
    CHECK(x2a_ <= x2b_) << "Size constraint";
    CHECK(y1a_ <= y1b_) << "Size constraint";
    CHECK(y2a_ <= y2b_) << "Size constraint";

    CHECK(x1b_ < x2a_) << "Size constraint";
    CHECK(y1b_ < y2a_) << "Size constraint";

  } else { 
    CHECK(x1a_ < x2a_) << "Size constraint";
    CHECK(y1a_ < y2a_) << "Size constraint";
  }
 
  CHECK((random_ && resample_) || (!random_))  << "the layer should be resampled if you use random crop";
  CHECK_GT(num_classes_, 0) << "num_classes should be larger than 0";
    
}


template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (resample_) {
    top[0]->ReshapeLike(bottom[0]);
    tmp_.ReshapeLike(bottom[0]);
  } else {
    std::vector<int> shape;
    shape.push_back(bottom[0]->num());
    shape.push_back(bottom[0]->channels());
    shape.push_back(y2a_ - y1a_);
    shape.push_back(x2a_ - x1a_);
    top[0]->Reshape(shape);
  }
}


template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype* crop_data;
  if (resample_) crop_data = tmp_.mutable_cpu_data();
  else crop_data = top[0]->mutable_cpu_data();

  int offset_oh, offset_oc, offset_on;
  int offset_ih, offset_ic, offset_in;
  int n, c, h, w;

  if (resample_) {
    offset_oh = tmp_.offset(0,0,1);
    offset_oc = tmp_.offset(0,1);
    offset_on = tmp_.offset(1);
  } else {
    offset_oh = top[0]->offset(0,0,1);
    offset_oc = top[0]->offset(0,1);
    offset_on = top[0]->offset(1);
  }

  offset_ih = bottom[0]->offset(0,0,1);
  offset_ic = bottom[0]->offset(0,1);
  offset_in = bottom[0]->offset(1);
  
  if (random_)
  {
    n = top[0]->num();
    c = top[0]->channels();
    h = 

    // cropping
    for (int i=0; i<num_; i++)
    {
      for (int j=0; j<num_classes_; j++)
      {
        top_data[i * num_classes_ + j] = (bottom_data[i] == j);
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if ( random_ ) return; // we don't calculate gradient if we use random crop 
}


#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS(CropLayer);

}  // namespace caffe

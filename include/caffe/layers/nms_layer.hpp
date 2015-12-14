
#ifndef CAFFE_NMS_LAYER_HPP_
#define CAFFE_NMS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class NMSLayer: public Layer<Dtype> {
 public:
  explicit NMSLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NMS"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // NMS layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int height_, width_, channels_;
  int kernel_top_, kernel_right_;
  int kernel_bottom_, kernel_left_;
  float activated_coeff_;
  float unactivated_coeff_;
  bool no_backprop_;
  Blob<Dtype> mask_;
};


}  // namespace caffe

#endif  // CAFFE_NMS_LAYER_HPP_

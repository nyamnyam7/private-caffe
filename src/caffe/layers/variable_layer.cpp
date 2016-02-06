#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/variable_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void VariableLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BlobShape& var_shape = this->layer_param_.variable_param().shape();
  const int num_axes = var_shape.dim_size();

  shape_.clear();
  for (int i = 0; i < num_axes; ++i) {
    const int sz = var_shape.dim(i);
    shape_.push_back(sz);
  }

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(shape_));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.variable_param().filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void VariableLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*this->blobs_[0]);
  top[0]->ShareData(*this->blobs_[0]);
  top[0]->ShareDiff(*this->blobs_[0]);
}

INSTANTIATE_CLASS(VariableLayer);
REGISTER_LAYER_CLASS(Variable);

}  // namespace caffe

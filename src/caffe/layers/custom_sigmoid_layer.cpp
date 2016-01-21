#include <cmath>
#include <vector>

#include "caffe/layers/custom_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void CustomSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    min_ = this->layer_param_.custom_sigmoid_param().min();
    max_ = this->layer_param_.custom_sigmoid_param().max();
}

template <typename Dtype>
inline Dtype custom_sigmoid(Dtype x, const Dtype min, const Dtype diff) {
  return min + diff / (1. + exp(-x));
}

template <typename Dtype>
void CustomSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype diff = max_ - min_;
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = custom_sigmoid(bottom_data[i], min_, diff);
  }
}

template <typename Dtype>
void CustomSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype diff = max_ - min_;

    for (int i = 0; i < count; ++i) {
      const Dtype custom_sigmoid_x = top_data[i];
      Dtype amp = custom_sigmoid_x - min_;
      bottom_diff[i] = top_diff[i] * amp * (1. - amp / diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CustomSigmoidLayer);
#endif

INSTANTIATE_CLASS(CustomSigmoidLayer);


}  // namespace caffe

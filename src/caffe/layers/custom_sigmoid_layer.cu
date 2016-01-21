#include <cmath>
#include <vector>

#include "caffe/layers/custom_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CustomSigmoidForward(const int n, const Dtype* in, Dtype* out, const Dtype min, const Dtype diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = min + diff / (1. + exp(-in[index]));
  }
}

template <typename Dtype>
void CustomSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype diff = max_ - min_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  CustomSigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, min_, diff);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void CustomSigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff, const Dtype min, const Dtype diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype custom_sigmoid_x = out_data[index];
    Dtype amp = custom_sigmoid_x - min;
    out_diff[index] = in_diff[index] * amp * (1 - amp / diff);
  }
}

template <typename Dtype>
void CustomSigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype diff = max_ - min_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    CustomSigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff, min_, diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CustomSigmoidLayer);


}  // namespace caffe

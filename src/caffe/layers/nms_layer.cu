#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
__global__ void NMSForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width, 
    const int kernel_h, const int kernel_w,
    Dtype* top_data,
    Dtype* mask) {
  // Iterate over top_data
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height / channels;
    int c = (index / width / height) % channels;

    int jstart = w - kernel_w;
    int jend = w + kernel_w+1;
    int istart = h - kernel_h;
    int iend = h + kernel_h+1;
    
    Dtype curval = bottom_data[index];
    bool is_maximum = true;
    // extremely inefficient implementation of non-maximum suprression
    const Dtype* rel = bottom_data + (n * channels + c) * height * width;
    if (jstart > 0 && jend <= width && istart > 0 && iend <= height)
    {
        for (int i=istart; i<iend; i++){
            for (int j=jstart; j<jend; j++){
                if ( rel[i * width + j] > curval ) is_maximum = false;
            }
        }
    }
    else is_maximum = false;

    if (is_maximum) {
        top_data[index] = bottom_data[index];
        mask[index] = 1.0;
    }
    else
    {
        top_data[index] = 0.0;
        mask[index] = 0.0;
    }
  }
}


template <typename Dtype>
void NMSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  Dtype* mask = NULL;

  if (use_top_mask) {
    mask = top[1]->mutable_gpu_data();
  } else {
    mask = mask_.mutable_gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  NMSForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, kernel_h_, kernel_w_,
      top_data, mask);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void NMSBackward(const int nthreads,
    const Dtype* top_diff, const Dtype* mask, Dtype* bottom_diff) {
  // Iterate over bottom_diff
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    if (mask[index] > 0) bottom_diff[index] = top_diff[index];
    else bottom_diff[index] = 0.0;
  }
}


template <typename Dtype>
void NMSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const Dtype* mask = NULL;
  if (use_top_mask) {
    mask = top[1]->gpu_data();
  } else {
    mask = mask_.gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  NMSBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mask, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(NMSLayer);


}  // namespace caffe

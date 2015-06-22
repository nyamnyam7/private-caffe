#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


/*

Note

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

*/

namespace caffe {

template <typename Dtype>
__global__ void MaxoutForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width, 
    const int pooled_channels,
    const int chgroup_sz,
    Dtype* top_data,
    int* mask, Dtype* top_mask) {
  // Iterate over top_data
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int pn = index / width / height / pooled_channels;
    int n = pn;
    int pc = (index / width / height) % pooled_channels;

    int cstart = pc * chgroup_sz; 
    int cend   = cstart + chgroup_sz; 

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;

    bottom_data += n * channels * height * width;
    for (int c = cstart; c < cend; ++c) {
      if (bottom_data[ (c * height + h) * width + w] > maxval) {
        maxidx =  (c * height + h) * width + w;
        maxval = bottom_data[maxidx];
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void MinoutForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width, 
    const int pooled_channels,
    const int chgroup_sz,
    Dtype* top_data,
    int* mask, Dtype* top_mask) {
  // Iterate over top_data
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int pn = index / width / height / pooled_channels;
    int n = pn;
    int pc = (index / width / height) % pooled_channels;

    int cstart = pc * chgroup_sz; 
    int cend   = cstart + chgroup_sz; 

    Dtype maxval = FLT_MAX;
    int maxidx = -1;

    bottom_data += n * channels * height * width;
    for (int c = cstart; c < cend; ++c) {
      if (bottom_data[ (c * height + h) * width + w] < maxval) { // inequlaity changed
        maxidx =  (c * height + h) * width + w;
        maxval = bottom_data[maxidx];
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}


template <typename Dtype>
void MaxoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  if (use_top_mask) {
    top_mask = top[1]->mutable_gpu_data();
  } else {
    mask = max_idx_.mutable_gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (!minout_)
    MaxoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, chgroup_sz_,
        top_data, mask, top_mask);
  else 
    MinoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_channels_, chgroup_sz_,
        top_data, mask, top_mask);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxoutBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_channels, const int chgroup_sz,
    Dtype* bottom_diff) {
  // Iterate over bottom_diff
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int pc = c / chgroup_sz; 
    int pn = n;

    const int pooled_idx  = (pc*height + h) * width + w;
    const int pooled_base = pn * pooled_channels * height * width;
    
    const int bottom_idx  = (c*height + h) * width + w;

    Dtype gradient = 0;
    top_diff += pooled_base;
    if (mask) {
      mask += pooled_base;
      if (mask[pooled_idx] == bottom_idx) {
        gradient += top_diff[pooled_idx];
      }
    } else {
      top_mask += pooled_base;
      if (top_mask[pooled_idx] == bottom_idx) {
        gradient += top_diff[pooled_idx];
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void MaxoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  if (use_top_mask) {
    top_mask = top[1]->gpu_data();
  } else {
    mask = max_idx_.gpu_data();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mask, top_mask, top[0]->num(), channels_,
      height_, width_, pooled_channels_, chgroup_sz_,
      bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(MaxoutLayer);


}  // namespace caffe

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
__global__ void OnehotForward(const int nthreads, const Dtype* bottom_data,
    Dtype* top_data,
    const int num,  
    const int outdim){
  // Iterate over top_data
  CUDA_KERNEL_LOOP(index, nthreads) {
    int outdim_n = index / outdim;
    int outdim_c = index % outdim;
    top_data[index] = (outdim_c == bottom_data[outdim_n]);
  }
}


template <typename Dtype>
void OnehotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  OnehotForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, bottom[0]->num(), num_classes_);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
void OnehotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
}


INSTANTIATE_LAYER_GPU_FUNCS(OnehotLayer);


}  // namespace caffe

#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (broadcast_) {
    const Dtype* bottom_data_a = NULL;
    const Dtype* bottom_data_b = NULL;
    int dima[4];
    int dimb[4];
    for (int i=0; i<4; i++)
    {
      dima[i] = bottom[0]->shape()[i];
      dimb[i] = bottom[1]->shape()[i];
    }
    bottom_data_a = bottom[0]->gpu_data();
    bottom_data_b = bottom[1]->gpu_data();

    switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      caffe_gpu_mul_broadcast<Dtype>(dima, dimb, bottom_data_a, bottom_data_b, top_data);
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      caffe_gpu_add_broadcast<Dtype>(dima, dimb, bottom_data_a, bottom_data_b, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown elementwise broadcast operation.";
    }
  } else {
    switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
          top_data);
      for (int i = 2; i < bottom.size(); ++i) {
        caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      caffe_gpu_set(count, Dtype(0.), top_data);
      // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
      for (int i = 0; i < bottom.size(); ++i) {
        caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      mask = max_idx_.mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
      for (int i = 2; i < bottom.size(); ++i) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
    }
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (broadcast_){
    bool broadcasted[2];
    int i, j;
    broadcasted[0] = broadcasted[1] = false;
    for (int i=0; i<4; i++) {
      if (bottom[0]->shape()[i] > bottom[1]->shape()[i]) broadcasted[1] = true;
      if (bottom[0]->shape()[i] < bottom[1]->shape()[i]) broadcasted[0] = true;
    }

    i=0; j=1; //i -> not broadcasted  j-> broadcasted
    if (broadcasted[0]){ i=1; j=0; }

    int dima[4], dimb[4];
    const Dtype* bot_data = bottom[i]->gpu_data();
    const Dtype* bot_data_brd = bottom[j]->gpu_data();
    Dtype*       bot_diff = bottom[i]->mutable_gpu_diff();
    Dtype*       bot_diff_brd = bottom[j]->mutable_gpu_diff();

    for (int n=0; n<4; n++) dima[n] = bottom[i]->shape()[n];
    for (int n=0; n<4; n++) dimb[n] = bottom[j]->shape()[n];

    switch(op_)
    {
    case EltwiseParameter_EltwiseOp_PROD:
      if (propagate_down[j]) {
        int n=0;
        for (int x=0; x<4; x++) n *= dima[x];
        caffe_gpu_mul<Dtype>(n, top_diff, bot_data, bot_diff);
        caffe_gpu_sum_reduce<Dtype>(dima, dimb, bot_diff, bot_diff_brd);
        caffe_gpu_set(n, Dtype(0), bot_diff);
      }
      
      if (propagate_down[i])
        caffe_gpu_mul_broadcast<Dtype>(dima, dimb, top_diff, bot_data_brd, bot_diff);

      break;
    case EltwiseParameter_EltwiseOp_SUM:
      if (propagate_down[j]) 
        caffe_gpu_sum_reduce<Dtype>(dima, dimb, top_diff, bot_diff_brd);

      if (propagate_down[i]) {
        int n=0;
        for (int x=0; x<4; x++) n *= dima[x];
        caffe_copy<Dtype>(n, top_diff, bot_diff);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
    }
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      if (propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        switch (op_) {
        case EltwiseParameter_EltwiseOp_PROD:
          if (stable_prod_grad_) {
            bool initialized = false;
            for (int j = 0; j < bottom.size(); ++j) {
              if (i == j) { continue; }
              if (!initialized) {
                caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
                initialized = true;
              } else {
                caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                              bottom_diff);
              }
            }
          } else {
            caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
          }
          caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
          break;
        case EltwiseParameter_EltwiseOp_SUM:
          if (coeffs_[i] == Dtype(1.)) {
            caffe_copy(count, top_diff, bottom_diff);
          } else {
            caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
          }
          break;
        case EltwiseParameter_EltwiseOp_MAX:
          mask = max_idx_.gpu_data();
          MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, top_diff, i, mask, bottom_diff);
          break;
        default:
          LOG(FATAL) << "Unknown elementwise operation.";
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe

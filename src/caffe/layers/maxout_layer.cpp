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
void MaxoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MaxoutParameter maxout_param = this->layer_param_.maxout_param();

  CHECK(maxout_param.has_chgroup_sz()) << "chgroup_sz should be defined";
  chgroup_sz_ = maxout_param.chgroup_sz();
  
  CHECK_GT(chgroup_sz_, 0) << "chgroup_sz should be larger than 0";
    
}


template <typename Dtype>
void MaxoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  CHECK_EQ(0, channels_ % chgroup_sz_) << "chgroup_sz parameter should divide the # of channels of"
    " the input";
  
  pooled_channels_ = channels_ / chgroup_sz_;

  // we will initialize the top blob and the vector index part.
  top[0]->Reshape(bottom[0]->num(), pooled_channels_, height_, width_);
  if (top.size()>1)
      top[1]->ReshapeLike(*top[0]);
  max_idx_.Reshape(bottom[0]->num(), pooled_channels_, height_, width_);
}


template <typename Dtype>
void MaxoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.

  // Initialize
  if (use_top_mask) {
    top_mask = top[1]->mutable_cpu_data();
    caffe_set(top_count, Dtype(-1), top_mask);
  } else {
    mask = max_idx_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
  }
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int pc = 0; pc < pooled_channels_; ++pc) {
          const int cstart = chgroup_sz_ * pc;
          const int cend = cstart + chgroup_sz_;

          for (int h = 0; h < height_; ++h) {
              for (int w = 0; w < width_; ++w) {
                  for (int c = cstart; c < cend; c++) {
                      const int index      = c  * width_ * height_ + h * width_ + w; 
                      const int pool_index = pc * width_ * height_ + h * width_ + w; 

                      if (bottom_data[index] > top_data[pool_index]) {
                          top_data[pool_index] = bottom_data[index];
                          if (use_top_mask) {
                              top_mask[pool_index] = static_cast<Dtype>(index);
                          } else {
                              mask[pool_index] = index;
                          }
                      }
                  }
              }
          }
      }
      bottom_data += bottom[0]->offset(1,0);
      top_data += top[0]->offset(1,0);
      if (use_top_mask) {
          mask += top[0]->offset(1,0);
      } else {
          top_mask += top[0]->offset(1,0);
      }
  }

}

template <typename Dtype>
void MaxoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;

  // The main loop
  if (use_top_mask) {
    top_mask = top[1]->cpu_data();
  } else {
    mask = max_idx_.cpu_data();
  }
  for (int n = 0; n < top[0]->num(); ++n) {
    const int size = top[0]->offset(1, 0);
    
    if (use_top_mask) {
        for (int pool_idx = 0; pool_idx < size; ++pool_idx) {
          bottom_diff[static_cast<int>(top_mask[pool_idx])] += top_diff[pool_idx];
        }
    } else {
        for (int pool_idx = 0; pool_idx < size; ++pool_idx) {
          bottom_diff[mask[pool_idx]] += top_diff[pool_idx];
        }
    }

    bottom_diff += bottom[0]->offset(1,0);
    top_diff += top[0]->offset(1,0);
    if (use_top_mask) {
      mask += top[0]->offset(1,0);
    } else {
      top_mask += top[0]->offset(1,0);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(MaxoutLayer);
#endif

INSTANTIATE_CLASS(MaxoutLayer);

}  // namespace caffe

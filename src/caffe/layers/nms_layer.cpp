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
void NMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NMSParameter nms_param = this->layer_param_.nms_param();

  CHECK(nms_param.has_kernel_left()) << "kernel_left should be defined";
  CHECK(nms_param.has_kernel_right()) << "kernel_right should be defined";
  CHECK(nms_param.has_kernel_top()) << "kernel_top should be defined";
  CHECK(nms_param.has_kernel_bottom()) << "kernel_bottom should be defined";
  kernel_top_ = nms_param.kernel_top();
  kernel_bottom_ = nms_param.kernel_bottom();
  kernel_left_ = nms_param.kernel_left();
  kernel_right_ = nms_param.kernel_right();
  activated_coeff_ = nms_param.activated_coeff();
  unactivated_coeff_ = nms_param.unactivated_coeff();

  no_backprop_ = nms_param.no_backprop() == 1;
  
  CHECK_GT(kernel_top_, -1) << "height should be larger than -1";
  CHECK_GT(kernel_left_, -1) << "width should be larger than -1";
  CHECK_GT(kernel_right_, -1) << "height should be larger than -1";
  CHECK_GT(kernel_bottom_, -1) << "width should be larger than -1";
}


template <typename Dtype>
void NMSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // we will initialize the top blob and the vector index part.
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
  if (top.size()>1)
      top[1]->ReshapeLike(*top[0]);
  mask_.Reshape(bottom[0]->num(), channels_, height_, width_);
}


template <typename Dtype>
void NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  Dtype* mask = NULL;  // suppress warnings about uninitalized variables
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.

  // Initialize
  if (use_top_mask) mask = top[1]->mutable_cpu_data();
  else  mask = mask_.mutable_cpu_data();

  caffe_set(top_count, Dtype(unactivated_coeff_), mask);
  // The main loop
  
  int hstep = ((kernel_top_ > kernel_bottom_)?kernel_top_:kernel_bottom_) + 1;
  int vstep = ((kernel_left_ > kernel_right_)?kernel_left_:kernel_right_) + 1;

  for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c=0; c < channels_; c++){
          // Inferior implementation of
          // block algorithm in "Efficient non-maximum suppression"
          for (int i=kernel_top_; i<height_-kernel_bottom_; i+=vstep) {
              for (int j=kernel_left_; j<width_-kernel_right_; j+=hstep) {
                  int mi = i, mj = j; // maximum
                  int mmi = i, mmj = j; // minimum
                  Dtype val_mi_mj = bottom_data[mi * width_ + mj];
                  Dtype val_mmi_mmj = bottom_data[mmi * width_ + mmj];

                  for (int i2 = i; i2<=i+kernel_bottom_; i2++) {
                      for (int j2 = j; j2<=j+kernel_right_; j2++) {
                          Dtype val_i2_j2 = bottom_data[i2 * width_ + j2];
                          if (val_i2_j2 > val_mi_mj) {
                              mi = i2;
                              mj = j2;
                              val_mi_mj = val_i2_j2;
                          }
                          if (val_i2_j2 < val_mmi_mmj) {
                              mmi = i2;
                              mmj = j2;
                              val_mmi_mmj = val_i2_j2;
                          }
                      }
                  }
         
                  bool max_failflag = false;
                  bool min_failflag = false;

                  for (int i2=mi-kernel_top_; i2<=mi+kernel_bottom_; i2++) {
                      for (int j2=mj-kernel_left_; j2<=mj+kernel_right_; j2++) {
                          Dtype val_i2_j2 = bottom_data[i2 * width_ + j2];
                          if (val_i2_j2 > val_mi_mj) {
                              max_failflag = true;
                              break;
                          }
                          if (max_failflag) break;
                      }
                      if (max_failflag) break;
                  }
                  
                  for (int i2=mmi-kernel_top_; i2<=mmi+kernel_bottom_; i2++) {
                      for (int j2=mmj-kernel_left_; j2<=mmj+kernel_right_; j2++) {
                          Dtype val_i2_j2 = bottom_data[i2 * width_ + j2];
                          if (val_i2_j2 < val_mmi_mmj) {
                              min_failflag = true;
                              break;
                          }
                          if (min_failflag) break;
                      }
                      if (min_failflag) break;
                  }
         
                  if (!max_failflag)
                  {
                      mask[mi * width_ + mj] = activated_coeff_;
                  }
                  if (!min_failflag)
                  {
                      mask[mmi * width_ + mmj] = activated_coeff_;
                  }
              }

          }

          for (int i=0; i<bottom[0]->offset(0, 1); i++)
              top_data[i] = bottom_data[i] * mask[i];

          bottom_data += bottom[0]->offset(0,1);
          top_data += top[0]->offset(0,1);
          mask += top[0]->offset(0,1);
      }
  }



}

template <typename Dtype>
void NMSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const Dtype* mask = NULL;  // suppress warnings about uninitialized variables

  // The main loop
  if (use_top_mask) {
    mask = top[1]->cpu_data();
  } else {
    mask = mask_.cpu_data();
  }

  if (!no_backprop_)
  {
    for (int n = 0; n < top[0]->num(); ++n) {
      const int size = top[0]->offset(1, 0);
      for (int i = 0; i < size; ++i) {
        bottom_diff[i] = mask[i] * top_diff[i];
      }
 
      bottom_diff += bottom[0]->offset(1,0);
      top_diff += top[0]->offset(1,0);
      mask += top[0]->offset(1,0);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NMSLayer);
#endif

INSTANTIATE_CLASS(NMSLayer);

}  // namespace caffe
